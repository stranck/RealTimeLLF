import sys
from PIL import Image

args = sys.argv
if(len(args) < 3):
	print(f"{args[0]} <input txt file> <output .png file>")
	exit(1)

f = open(args[1], "r")
lines = f.readlines()
f.close()

while(not lines[0].startswith("STATIC_IMG_DIMENSIONS: ")):
	lines = lines[1:]
l0 = lines[0].strip().split()
width = int(l0[1])
height = int(l0[2])
print(width, height)
img = Image.new('RGBA', (width, height))

lines = lines[1:]
data = []
print(f"Lines count: {len(lines)}")
for l in lines:
	currentData = bytes.fromhex(l)
	byteIdx = 0
	if(len(currentData)/4 != width):
		print(f"Got row of {len(currentData)/4} pixels, expected {width}") 
	while(byteIdx < len(currentData)):
		data.append((currentData[byteIdx + 0], currentData[byteIdx + 1], currentData[byteIdx + 2], currentData[byteIdx + 3]))
		byteIdx += 4

print(f"Expected len: {width * height} \tGot len: {len(data)}")
img.putdata(data)
img.save(args[2], "PNG")