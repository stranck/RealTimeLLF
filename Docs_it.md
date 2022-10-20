# Laplacian Filter in CUDA & OpenMp
_Luca Sorace_
_Lorenzo P._

Il laplacian filter è un filtro capace di fare tone mapping e di aumentare o diminuire i dettagli di un'immagine senza creare né aloni né distorsioni. È basato sulle piramidi laplaciane e gaussiane: crea una sottopiramide molto piccola dell’immagine originale ed applica la correzione solamente a quella piccola porzione.

- [Sito dello studio](https://people.csail.mit.edu/sparis/publi/2011/siggraph/)
- [Paper ufficiale](https://people.csail.mit.edu/sparis/publi/2011/siggraph/Paris_11_Local_Laplacian_Filters.pdf)

L’algoritmo è diviso in tre grandi fasi:

## Fase 1
- Creo una piramide gaussiana $inputGauss$ dall’immagine originale $inImg$ (La definizione di una piramide gaussiana si trova  nella [sezione dedicata](#Gaussiana))
## Fase 2
Per ogni pixel $G_0$, per ogni livello in $inputGauss$:
- Taglio una sottoregione $R_0$ intorno a $G_0$ da $inImg$, di grandezza proporzionale alla dimensione del livello corrente
- Applico a $R_0$ una funzione di remapping usando $G_0$ come reference
- Creo una piramide gaussiana $tempGauss$ su $R_0$
- Creo una piramide laplaciana $L_{l0}$ a partire da $tempGauss$ (La definizione di una piramide laplaciana si trova  nella [sezione dedicata](#Laplaciana))
- Prendo il pixel con stesse coordinate di $G_0$ da $L_{l0}$ e lo copio sulla piramide laplaciana $outputLaplacian$
## Fase 3
- Copio il livello più piccolo di $inputGauss$ nel livello più piccolo di $outputLaplacian$
- Collasso $outputLaplacian$ utilizzando la seguente formula: `for(n = nLevels -> 1) lapl[n - 1] = lapl[n - 1] + upsample(lapl[n])`. L'output sarà in `lapl[0]` e verrà copiato sull'immagine di destinazione $outImg$
- Applico una funzione di clamp su $outImg$

![](https://i.imgur.com/3Kq0XIL.png)

# Strutture Dati
## Image 
Abbiamo due tipi di immagini: Image3 e Image4, le quali differiscono dal tipo di pixel che contengono (Pixel3 per le immagini RGB, o Pixel4 per le immagini RGBA). I due tipi di immagini sono definiti come segue:
```c=

typedef struct {
    uint32_t height; //altezza dell’immagine.
    uint32_t width;  //larghezza dell’immagine.
    Pixel *pixels;   //buffer dei pixel dell’immagine.
} Image;
```

## Vects 
Abbiamo molti tipi di vettori che variano in base al tipo ed alla quantità di variabili che contengono. Due esempi sono:
```c= 
typedef struct {
    float x;
    float y;
    float z;
    float w;
} Vec4f;
typedef struct {
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t w;
} Vec4u8;
```
In generale, la naming convention è la seguente:
```
Vec          -> è un vettore
[3|4]        -> numero di variabili del vettore
[u|i|f]      -> tipo di variabili del vettore (unsigned int, int, float)
[8|16|32|64] -> dimensione in bit delle variabili del vettore
```

## WorkingBuffers
I working buffers sono delle strutture dati preallocate che contengono tutti i buffer utilizzati all'interno del rendering, in modo che non debbano essere riallocati ad ogni iterazione.

Ogni implementazione di LLF ha i suoi WorkingBuffer: 
### Single core _(llf)_
```c=
typedef struct {
	Kernel filter;                  //Kernel per il blur già precalcolato
	Pyramid gaussPyramid;           //Piramide gaussiana che viene creata nella fase 1 dell'algoritmo
	Pyramid outputLaplacian;        //Piramide laplaciana da collassare che viene calcolata nella fase 2 dell'algoritmo
	Pyramid bufferGaussPyramid;     //Piramide gaussiana temporanea utilizzata all'interno della fase 2 dell'algoritmo
	Pyramid bufferLaplacianPyramid; //Piramide laplaciana temporanea utilizzata all'interno della fase 2 dell'algoritmo
} WorkingBuffers;
```
### OpenMP
```c=
typedef struct {
	uint32_t end;             //Numbero di pixel totali presenti dentro gaussPyramid, ovvero il numero di pixel totali da renderizzare nella fase 2
	Kernel filter;            //Kernel per il blur già precalcolato
	Pyramid *bArr;            //Array di piramidi gaussiane temporanee utilizzate, una da ogni thread differente, all'interno della fase 2 dell'algoritmo
	uint32_t *pyrDimensions;  //Numero di pixel precalcolato per ogni livello della gaussPyramid, ovvero il numero di pixel per renderizzare ogni singolo livello differente della gaussPyramid nella fase 2
	CurrentLevelInfo *cliArr; //Array di CurrentLevelInfo (vedi più avanti nella sezione dedicata), uno per ogni thread
	Pyramid gaussPyramid;     //Piramide gaussiana che viene creata nella fase 1 dell'algoritmo
	Pyramid outputLaplacian;  //Piramide laplaciana da collassare che viene calcolata nella fase 2 dell'algoritmo
} WorkingBuffers;
```
### CUDA
```c=
typedef struct {
	Kernel d_filter;           //Kernel per il blur già precalcolato salvato sulla memoria globale del device
	Pyramid d_gaussPyramid;    //Piramide gaussiana salvata sulla memoria globale del device che viene creata nella fase 1 dell'algoritmo
	Pyramid d_outputLaplacian; //Piramide laplaciana da collassare, salvata sulla memoria globale del device, che viene calcolata nella fase 2 dell'algoritmo
	Image3 *d_img;             //Immagine di buffer salvata sulla memoria globale del device usata come input/output per la parte cuda dell'algoritmo 
} WorkingBuffers;
```

## CurrentLevelInfo
Questa struttura è utilizzata solo per l'implementazione in openmp.
Contiene informazioni sul livello corrente nella fase 2 dell'algoritmo, assieme ad altri dati in cache
```c=
typedef struct {
	uint8_t currentNLevels;         //Numero di livelli presenti nelle piramidi buffer che utilizziamo al livello corrente durante la fase 2
	uint8_t lev;                    //Numero del livello corrente durante la fase 2
	uint32_t oldY;                  //coordinata Y attuale nell'elaborazione del livello corrente durante la fase 2
	uint32_t width;                 //larghezza del livello corrente durante la fase 2
	uint32_t prevLevelDimension;    //Numero di pixel che abbiamo renderizzato fino all'inizio del livello corrente durante la fase 2
	uint32_t nextLevelDimension;    //Numero di pixel che avremo renderizzato alla fine del livello corrente durante la fase 2. Viene usato per capire quando si deve passare all'elaborazione del livello successivo
	uint32_t subregionDimension;    //Dimensioni della sottoregione R0 nel livello corrente durante la fase 2
	uint32_t full_res_roi_yShifted; //Valore precalcolato di full_res_roi_yShifted
	uint32_t base_y;                //Valore precalcolato di base_y
	uint32_t end_y;                 //Valore precalcolato di end_y
	Image3 *currentGaussLevel;      //Puntatore al livello corrente della piramide gaussiana in input
} CurrentLevelInfo;
```

# Typedef
## Pyramid
`typedef Image3** Pyramid;`

La piramide è un array di immagini le cui dimensioni si riducono quadraticamente ad ogni livello. Il livello zero è quello più grande.
```
lev[0].width = imgSource.width
lev[n].width = lev[n - 1].width / 2

Stessa cosa per l'height
```
Esistono due tipi di piramidi, definiti come segue: 
### Gaussiana
```
gauss[0] = sourceImg
gauss[n] = downsample(gauss[n - 1])

Dove

downsample(img) è una funzione che dimezza e applica un kernel di blur sull’input.
```
### Laplaciana
```
lap[nLevels] = gauss[nLevels]
lap[n] = gauss[n] - upsample(gauss[n + 1])

Dove

upsample(img) è una funzione che duplica e applica un kernel di blur all’input.
```

## Pixel
`typedef Vec4f Pixel4;`

I pixel sono dei [vettori](#Vects), in cui xyzw rappresentano i valori RGBA

## AlphaMap
`typedef uint8_t* AlphaMap;`

Una alpha map è una matrice di `uint8_t` in cui in ogni posizione viene salvato il canale alpha del corrispettivo pixel di una Image3


## Kernel
`typedef float* Kernel;`

Un Kernel è una matrice di `float` con valori compresi tra 0 ed 1, utilizzata nella funzione `convolve(image, kernel)`, definita come segue:

Per ogni pixel $C$ dell'immagine in input $I$:
- Prendiamo una sottoregione $R$ da $I$ di dimensioni $KERNEL\_DIMENSION^2$ con centro in $C$
- Moltiplichiamo ogni pixel in $R$ con il corrispondente valore del $kernel$
- Sommiamo tutti i pixel moltiplicati
- Salviamo il risultato nel pixel alle stesse coordinate di $C$ nell'immagine in output


All'interno dell'algoritmo LLF il kernel viene utilizzato per applicare un filtro di blur nelle fasi di upsampling e downsampling, usate per costruire le piramidi gaussiane e laplaciane

# Singlecore
La versione singlecore utilizza l’algoritmo descritto inizialmente, ma con la piccola ottimizzazione dove downsampling e upsampling vengono svolte contemporaneamente al convolving. In questo modo risparmiamo tempo evitando copie extra delle immagini e non abbiamo neanche bisogno di ulteriori buffer durante le fasi intermedie di upsample e downsample.
Le funzioni originali sono comunque riportate nello stesso file.

![](https://i.imgur.com/OApxh3Z.png)

_Risultato dell'implementazione single core_

## Performance

| Image Size    | Time in ms |
| ------------- |:---------- |
| **480x320**   | 5815       |
| **800x533**   | 16612      |
| **1920x1279** | 98914      |
| **3840x2558** | 384384     |

_Misure in ms. Test svolti con processore AMD r5 5600x_

# OpenMP
## Idea algoritmo
Durante il rendering del filtro LLF non abbiamo dipendenze tra i singoli pixel che stiamo renderizzando, bensì abbiamo dipendenze tra i differenti layer delle piramidi.

Durante la fase 1 e 3 dell'algoritmo andiamo a parallelizzare (durante le operazioni di creazione della piramide gaussiana e di collaso di quella laplaciana) solamente il rendering di ogni singolo layer: ogni thread lavorerà sul suo set di pixel indipendentemente dagli altri e tra un layer e l'altro avverrà una sincronizzazione.

Durante la fase 2, invece, non esiste dipendenza né tra layer né tra pixel, e quindi la parallelizzazione avviene renderizzando più pixel contemporaneamente senza alcun bisogno di sincronizzazione. Il rendering di ciascun pixel avverrà analogalmente alla precedente implementazione single core. Ogni thread avrà una propria struttura dati contenente i suoi buffer personali per il rendering, più informazioni e dati "in cache" sul livello corrente che sta renderizzando. Lavorerà indipendentemente da tutti gli altri thread. 

All'interno della fase 2 viene applicata una prima ottimizzazione sostanziale a livello di algoritmo: poiché dell'intera piramide laplaciana temporanea ci interessa solamente un pixel (quello che poi verrà copiato nella piramide di output), al posto di renderizzarla tutta, verrá upsampleata e blurrata solo la sezione che lo determina.

## Architettura
Rispetto alla versione single core, vengono effettuate delle operazioni in più:

**FASE 1:**
- Vengono allocati e precalcolati nei workingBuffers:
    - la dimensione di ogni livello della piramide gaussiana in input
    - il numero di pixel totali da renderizzare
    - una struttura dati (CurrentLevelInfo) contenente le informazioni e valori precalcolati sul livello corrente nella fase 2
    - un array contenente una piramide gaussiana di buffer per thread, che verrà usata nella fase 2
- viene calcolata la piramide gaussiana di input parallelizzando la costruzione di ogni singolo layer: all'interno di `downsampleConvolve_parallel()` viene lanciata una parallel for che scorre ogni pixel del layer corrente
- Ogni thread inizializza la propria struttura `currentLevelInfo` al livello 0 della piramide gaussiana di input

**FASE 2:**
Viene lanciata un'unica parallel for che itererà su tutti i pixel di tutti i livelli della piramide gaussiana in input. Poiché **solo in questo caso** non c'è dipendenza tra i layer della piramide, non serviranno primitive di sincronizzazione tra l'uno e l'altro.

Per ogni pixel:
- Ogni thread usa il proprio id per prendere la sua piramide gaussiana di buffer ed il suo oggetto `currentLevelInfo` contenente le informazioni sul livello corrente che sta elaborando
- Controlliamo se abbiamo finito di lavorare sul livello corrente, accertando che l'id della parallel for sia più piccolo di `nextLevelDimension`, ovvero verificando se l'id della parallel for si trova ancora nel livello corrente, oppure fa parte del successivo
    - Se abbiamo finito di lavorare sul livello corrente, chiamiamo  `updateLevelInfo()`, che si occuperà di:
        - Aggiornare il numero del livello corrente
        - Aggiornare il valore precalcolato del numero di livelli nella piramide gaussiana temporanea
        - Aggiornare il valore precalcolato della dimensione della sottoregione
        - Salvare il numero di pixel che abbiamo renderizzato fin'ora e quanti ne dobbiamo calcolare per finire il nuovo livello (`nextLevelDimension`)
- si calcola `localIdx` come la differenza tra l'idx della parallel for ed il numero di pixel elaborati fino al livello precedente
- si ottengono la x e la y a partire da `localIdx` e si controlla se siamo passati a lavorare su un'altra riga del livello corrente della piramide gaussiana di input
    - Se stiamo lavorando su una nuova riga rispetto a prima (cambia il valore delle y), aggiorna i relativi valori salvati in cache all'interno di `currentLevelInfo`
- Si procede normalmente come nella implementazione singlecore, eccezion fatta della grossa ottimizzazione specificata nell'idea dell'algoritmo

**FASE 3:**
- Viene copiato in maniera parallela l'ultimo layer della piramide gaussiana di input sulla piramide laplaciana di output: più thread lavorano contemporaneamente alla copia dei pixel
- Viene collassata la piramide laplaciana in output usando più thread sia durante l'upsampling (analogalmente al downsampling della fase 1), che quando addizioniamo i singoli pixel dei due layer
- Viene clampata l'immagine in maniera parallela, analoga alla copia

## Scelte progettuali

La creazione in fase progettuale della struttura dati `currentLevelInfo` viene dall'idea di voler ridurre al minimo l'overhead della creazione di nuovi thread. Essa è stata creata proprio per fare in modo che ci sia un'unica parallel for, piuttosto che una per ogni layer (che invece sarebbe stata una scelta più "pulita" a livello di codice). La creazione di una tale struttura dati è stata possibile grazie al fatto che non esiste alcun tipo di dipendenza tra pixel nell'elaborazione della piramide laplaciana di output. Questa caratteristica, inoltre, ha semplificato di molto il processo di parallelizzazione e della scrittura di codice, permettendoci di applicare la scelta più ovvia: al posto di parallelizzare il lavoro per renderizzare un singolo pixel della piramide gaussiana di input, lavoriamo su più pixel contemporaneamente. In questo modo, tra l'altro, siamo sempre sicuri che il carico di ogni thread sarà equo rispetto agli altri

## Test di correttezza

Abbiamo testato l'algoritmo in openmp con vari parametri e abbiamo sempre riscontrato risultati visibilmente identici alla versione single core.

![](https://i.imgur.com/IsT61Te.png)

_Risultato dell'implementazione con openMP_

## Performance

| Image Size    | 24 Threads | 16 Threads | 12 Threads | 8 Threads | 4 Threads | 2 Threads | Single core _(Con stessa implementazione)_ |
|:------------- | ---------- |:---------- | ---------- |:--------- | --------- | --------- |:------------------------------------------ |
| **480x320**   | 70         | 87         | 146        | 168       | 279       | 443       | 872                                        |
| **800x533**   | 205        | 257        | 394        | 473       | 826       | 1301      | 2551                                       |
| **1920x1279** | 1213       | 1528       | 2271       | 2845      | 4881      | 7779      | 15344                                      |
| **3840x2558** | 4904       | 6226       | 9266       | 11338     | 18299     | 30788     | 60654                                      |

_Misure in ms. Test svolti con processore AMD r9 5900x_

Possiamo notare come, anche con input piccoli, lo speedup e l'efficenza rimangano costanti a parità di thread.
L'efficenza inversamente proporzionale al numero di thread indica che la gestione della parallelizzazione fa da collo di bottiglia alla velocità di esecuzione del codice

### Speedup

| Image Size    | 24 Threads | 16 Threads | 12 Threads | 8 Threads | 4 Threads | 2 Threads |
|:------------- |:---------- |:---------- | ---------- |:--------- | --------- |:--------- |
| **480x320**   | 12,4571    | 10,023     | 5,9726     | 5,1905    | 3,1254    | 1,9684    |
| **800x533**   | 12,4439    | 9,9261     | 6,4746     | 5,3932    | 3,0884    | 1,9608    |
| **1920x1279** | 12,6496    | 10,0419    | 6,7565     | 5,3933    | 3,1436    | 1,9725    |
| **3840x2558** | 12,3683    | 9,7420     | 6,5459     | 5,3496    | 3,3146    | 1,9701    |

### Efficienza

| Image Size    | 24 Threads | 16 Threads | 12 Threads | 8 Threads | 4 Threads | 2 Threads |
|:------------- |:---------- |:---------- | ---------- |:--------- | --------- |:--------- |
| **480x320**   | 0,5190     | 0,6264     | 0,4977     | 0,6488    | 0,7814    | 0,9842    |
| **800x533**   | 0,5185     | 0,6204     | 0,5396     | 0,6742    | 0,7721    | 0,9804    |
| **1920x1279** | 0,5271     | 0,6276     | 0,5630     | 0,6742    | 0,7859    | 0,9862    |
| **3840x2558** | 0,5153     | 0,6089     | 0,5455     | 0,6687    | 0,8287    | 0,9850    |

# Cuda
## Idea algoritmo
Durante la fase 1 e 3 dell'algoritmo i kernel cuda vengono lanciati con un solo blocco e più thread (La motivazione è nelle scelte progettuali) che elaboreranno più pixel, creando così una associazione 1:N tra thread e pixel da elaborare. Per mantenere le dipendenze tra layer, si usa una `__syncthreads()` alla fine di ogni upsample o downsample. L'unica funzione che fa eccezione è `d_clampImage3()`, dove invece c'è una associazione 1:1 tra thread e pixel da renderizzare

Durante la fase 2, invece, useremo più blocchi. Ciascun blocco si occuperà di renderizzare più pixel della piramide gaussiana di input, e ciascun thread del blocco si occuperà di upsampleare o downsampleare più pixel della relativa sottoregione. Creiamo così una associazione 1:N tra blocco e pixel della piramide gaussiana di input da renderizzare, ed un'altra associazione 1:N tra ogni thread (di un blocco) e il pixel della sottoregione da renderizzare. Così facendo stiamo applicando una doppia parallelizzazione: Renderizziamo più pixel della piramide gaussiana di input contemporaneamente, e parallelizziamo il rendering di ciascuno di essi.

Durante l'elaborazione del singolo pixel applichiamo una ottimizzazione a livello spaziale del miglioramento sostanziale applicato già nel codice relativo ad OpenMP (Quarto paragrafo dell'_"Idea algoritmo"_): quando calcoliamo la piramide gaussiana temporanea, ci salviamo solamente gli ultimi due layer più piccoli (Quelli utilizzati successivamente dalla versione già ottimizzata della piramide laplaciana). In questo modo possiamo avere entrambi i layer in shared memory ed ottenere così accessi molto più rapidi in lettura e scrittura durante i vari punti della fase 2, raggiungendo così uno speedup di circa il 60%

A differenza di openmp, in cuda siamo costretti a chiamare più volte il kernel della llf, una volta per ogni layer della piramide gaussiana di input

## Architettura
Rispetto all'implementazione single core, sono state riscritte molte funzioni di utility generale appositamente per cuda. In particolare:
- Le funzioni `min`, `max`, `clamp`, `smoothstep` sono state riscritte per ridurre al minimo la branch divergence utilizzando la formula `int b = <boolean formula>; return trueValue * b + falseValue * (1 - b)` (Questa formula verrà usata spesso per ridurre la branch divergence)
- Le funzioni `getPixel3` e `setPixel3` adesso si prendono già in input il buffer di pixel e la larghezza dell'immagine, per risparmiare accessi in global memory per leggere i campi dai metadati dell'immagine
- Tutte le funzioni di allocazione e distruzione di strutture dati sul device dall'host sono state riscritte per impostare i campi delle varie strutture sullo stack dell'host e successivamente copiare l'intera struttura già inizializzata sull'heap del device

Inoltre sono state fatte ulteriori modifiche alle 3 fasi dell'algoritmo:

**FASE 1:**
- Prima di ogni cosa, viene copiata l'immagine dalla memoria dell'host a quella del device
- La piramide gaussiana in input viene elaborata attraverso un kernel chiamato con un unico blocco e più thread (il motivo è spiegato in scelte progettuali). La suddivisione del lavoro sui vari thread è fatta analogalmente all'implementazione in openmp. L'unica differenza è che l'upsample convolve è stata leggermente modificata per ridurre la branch divergence

**FASE 2:**
Nella fase 2 l'host looppa per ogni layer della piramide gaussiana di input. Per ciascuno precalcola la dimensione della sottoregione, e chiama il kernel interno sul device utilizzando N blocchi ed M thread, dove `N = numero di pixel della piramide gaussiana di input che verranno elaborati contemporaneamente` e `M = numero di thread usati contemporaneamente per renderizzare un singolo pixel della piramide gaussiana di input`
Ciascun blocco:
- Allocherà in shared memory e ci copierà sopra il kernel di blur dalla global memory
- Allocherà in shared memory due buffer interscambiabili per l'elaborazione delle piramidi gaussiane e laplaciane interne
- Per ogni pixel che tale blocco deve renderizzare:
    - Ottieni il pixel $G_0$ usando un solo thread e salvalo in shared memory
    - Ritaglia ed applica la funzione di remapping contemporaneamente per risparmiare scritture/letture in memoria sul device. Ogni thread si occuperà di un batch di pixel da renderizzare. All'interno della funzione di remap sono state applicate delle ottimizzazioni per ridurre la branch divergence
    - Creo gli ultimi due layer della piramide gaussiana utilizzando solamente i due buffer in shared memory, scambiandoli ad ogni iterazione in modo da avere l'immagine più piccola sempre nello stesso buffer. Il calcolo della piramide gaussiana è parallelizzato nello stesso modo della fase 1
    - Utilizzo gli ultimi due layer che abbiamo appena calcolato per trovarmi il singolo pixel upsampleato da posizionare nella piramide laplaciana in output. Come in openmp, qui facciamo upsample solamente della regione di immagine da cui il pixel in output dipende e, siccome a differenza di openmp abbiamo più thread a disposizione, andiamo dapprima ad upsampleare e blurrare usando un thread diverso per ogni pixel (Quindi creando una associazione 1:1 tra thread e pixel), e poi utilizziamo una sum reduction per ottenere il valore finale a partire dai precedenti parziali
    - Solo un thread piazza quindi questo pixel nella piramide laplaciana di output 

**FASE 3:**
- Copia l'ultimo livello della piramide gaussiana di input sulla piramide laplaciana di output, utilizzando un blocco ed N threads (stesse ragioni di sopra): ogni thread copierà solo un piccolo batch di pixel
- Collassa la piramide laplaciana di output utilizzando un blocco composto da N thread, ciascuno si occuperà di renderizzare un batch di pixel di ogni layer, analogalmente ad openmp
- Clampa l'immagine utilizzando più blocchi e più threads, in modo da avere una vera associazione 1:1 tra thread e pixel da clampare

## Scelte progettuali
Il motivo per cui nelle fasi 1 e 3 lanciamo i kernel (che elaborano piramidi) con un solo blocco è in realtà molto semplice: poiché la dimensione delle piramidi si riduce quadraticamente ci si ritrova sempre con più thread che pixel da elaborare, rendendo inutile il lancio e sincronizzazione di più blocchi. Inoltre, poiché c'è una dipendenza tra i vari layer della piramidi, avremmo dovuto lanciare ogni volta un kernel per elaborare un singolo layer per mantenere la sincronizzazione necessaria, invece di avviarne soltanto uno all'inizio.
`d_clampImage3()` invece viene lanciato con più blocchi perché lavora su tutti i pixel dell'immagine di partenza, ben più dei thread possibili, ed in più non lavora con layer differenti dove sarebbero state necessarie primitive di sincronizzazione

La ragione per cui, a differenza di openmp, chiamiamo una volta per layer il kernel interno della llf, è perché avere thread e blocchi a lavorare contemporaneamente su layer differenti aumenterebbe la branch divergence: le "grandezze" dei for sono tutte uguali solo all'interno di uno stesso layer. L'esempio più immediato è applicare il downsample sulla sottoregione: su layer differenti la sottoregione avrà dimensioni differenti, di conseguenza i for all'interno del downsample avranno "durate" differenti.

Nella fase 2 NON facciamo una associazione 1:1 tra blocco e pixel da renderizzare per diminuire i tempi di accesso in memoria: Ad inizio del kernel interno della llf viene effettuata una copia dalla global alla shared memory del kernel di blur, in modo che poi ci si possa accedere più rapidamente nelle fasi di upsampling e downsampling. Avere un numero di blocchi troppo alto, vorrebbe dire effettuare questa copia troppe volte, rallentando di fatto l'esecuzione. Il numero di blocchi ideale corrisponde di conseguenza al numero di blocchi massimo che il device riesce a mandare in esecuzione contemporaneamente.


## Test di correttezza
![](https://i.imgur.com/tSyyNHz.png)

_Risultato dell'implementazione con CUDA_

## Performance

| Image Size    | 1Block 32Threads | 16Block 64Threads | 32Block 128Threads | 256Block 256Threads | 512Block 256Threads | 512Block 512Threads | OpenMP 1thread |
| ------------- | ---------------- |:----------------- | ------------------ | ------------------- |:------------------- |:------------------- | -------------- |
| **480x320**   | 1338             | 73                | 32                 | 21                  | 17                  | 30                  | 872            |
| **800x533**   | 3749             | 201               | 83                 | 52                  | 51                  | 84                  | 2551           |
| **1920x1279** | 22331            | 1091              | 452                | 279                 | 285                 | 461                 | 15344          |
| **3840x2558** | 95493            | 4431              | 1754               | 1112                | 1045                | 1763                | 60654          |

_Misure in ms. Test svolti con gpu RTX 3600_

Possiamo notare che, come accadeva esattamente con openmp, lo speedup rimane costante a parità di blocchi/threads.
Notiamo anche però come, superato una soglia di blocchi, lo speedup aumenti di pochissimo, poiché avremo più blocchi rispetto a quanti la gpu  riesca a eseguirne contemporaneamente e poiché dovremmo effettuare più copie del kernel di blur dalla global memory alla shared memory.
Aumentare di troppo il numero di thread, invece, è proprio controproducente. Questo perché con troppi thread abbiamo un numero esagerato di thread "a non fare niente" per le elaborazioni delle sottoregioni più piccole, ed inoltre riduciamo anche il numero di blocchi che verranno eseguiti contemporaneamente 

### Speedup

| Image Size    | 1Block 32Threads | 16Block 64Threads | 32Block 128Threads | 256Block 256Threads | 512Block 256Threads | 512Block 512Threads |
| ------------- | ---------------- |:----------------- | ------------------ |:------------------- |:------------------- |:------------------- |
| **480x320**   | 0,6517           | 11,9452           | 27,2500            | 41,5238             | 51,2941             | 29,0667             |
| **800x533**   | 0,6804           | 12,6915           | 30,7349            | 49,0577             | 50,0196             | 30,3690             |
| **1920x1279** | 0,6871           | 14,0642           | 33,9469            | 54,9964             | 53,8386             | 33,2842             |
| **3840x2558** | 0,6352           | 13,6886           | 34,5804            | 54,5450             | 58,0421             | 34,4039             |

# Implementazione real time con NDI®

Utilizzando NDI® (Nuovo standard industriale per la trasmissione real time di video a bassa compressione over IP, successore di SDI) abbiamo creato un'applicazione capace di ricevere una stream video e di inoltrare il traffico renderizzandoci sopra il filtro llf in tempo reale. La codebase è unica per tutte e tre le versioni; vengono usate flag del preprocessore per chiamare le funzioni corrette. 


_NDI® is a registered trademark of Vizrt Group._

## Algoritmo
Le prime cose che accadono all'avvio del programma sono il caricamento degli argomenti, il set dell'handler per il sigint e l'inizializzazione della libreria NDI®.
Dopo aver trovato la source NDI® col nome corrispondente all'apposito argomento, ci si connette specificando di voler utilizzare RGBX/A come formato colore (Di default viene usato UYVY), si crea il receiver ed il sender NDI®, e si avvia un secondo thread che si occuperà del rendering vero e proprio.

Il secondo thread inizialmente si occuperà di allocare momentaneamente "al minimo" i suoi buffer interni, i semafori ed inizializzare i working buffer della versione di LLF selezionata. A questo punto il programma entra in uno stato di working, riassumibile così:

**THREAD PRINCIPALE:**
- Si mette in ascolto di un pacchetto NDI®
- Appena ricevuto, se è un frame video continua, altrimenti torna all'inizio
- Passa il frame ricevuto al processing thread:
    - Entra in una sezione critica
    - Si salva le dimensioni del frame ricevuto
    - Se il frame è più grande di tutti i frame ricevuti fin'ora, rialloca i buffer interni
    - Copia il frame ricevuto sul buffer `mainThread->processingThread`
    - Esce dalla sezione critica
    - Segnala al processing thread che un nuovo frame è disponibile, utilizzando il semaforo `frameAvailable`
- Prende l'ultimo frame renderizzato dal processing thread:
    - Entra in una sezione critica
    - Setta le dimensioni del frame NDI® a quelle dell'ultimo frame renderizzato dal processingThread
    - Copia il contenuto del buffer `processingThread->mainThread` sul frame NDI® (Prendendo quindi i dati dell'ultimo frame renderizzato fin'ora dal processing thread)
    - Esce dalla sezione critica
- Lo spedisce a tutti i ricevitori NDI® in asoclto
- Controlla se è stato richiesto lo shutdown del programma, in caso chiama la funzione `cleanup()`

**PROCESSING THREAD:**
- Attende che un nuovo frame sia disponibile aspettando il semaforo `frameAvailable`
- Copia l'ultimo frame ricevuto dal thread principale, sull'immagine da renderizzare:
    - Entra in una sezione critica
    - Copia le dimensioni dell'immagine
    - Se le dimensioni dell'immagine sono più grandi di quelle di tutti i frame ricevuti fin'ora, rialloca l'immagine ed i workingBuffers della versione di llf scelta
    - Copia i dati dell'ultima immagine ricevuta dal buffer `mainThread->processingThread` e contemporaneamente effetta la condizione da `Pixel4u8` a `Pixel4f`
    - Esce dalla sezione critica
- Effetta il rendering del filtro llf usando la versione scelta in fase di preprocessing e ne attende il completamento
- Copia indietro l'immagine renderizzata:
    - Entra in una sezione critica
    - Si salva le dimensioni dell'immagine appena renderizzata
    - Copia i dati dall'immagine al buffer `processingThread->mainThread`, effettuando contemporaneamente la conversione da `Pixel4f` a `Pixel4u8`
    - Esce dalla sezione critica
- Controlla se la funzione `cleanup()` ha richiesto uno shutdown del processing thread