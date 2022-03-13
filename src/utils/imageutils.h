#ifndef STRUCTS_DEP
    #include "structs.h"
#endif

#ifndef IMAGE_UTILS_DEP
#define IMAGE_UTILS_DEP

Image makeImage(uint32_t width, uint32_t height);
Image makeImage(uint32_t width, uint32_t height, Pixel pixels[]);

inline Pixel * getPixel(Image img, uint32_t x, uint32_t y);
inline Pixel * getPixel(Image img, Vec2u32 v);

void destroyImage(Image img);

#endif