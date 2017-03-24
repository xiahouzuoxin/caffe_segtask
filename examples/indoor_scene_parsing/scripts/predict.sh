
IMG=$1

# 63 class
# python predict.py \
# 	--model=./test_pspnet50_ade20k64cls_v2.prototxt \
# 	--weights=./snapshots/pspnet50-ADE20K64cls_v2_iter_60000.caffemodel \
# 	--clstxt=ADE20K/classes_63.txt ${IMG}

python predict.py \
	--model=./test_pspnet50_ade20k64cls_v2.prototxt \
	--weights=./snapshots/pspnet50-ADE20K64cls_v2_iter_60000.caffemodel \
	--clstxt=ADE20K/classes_63.txt \
	--outdir=./results/showroom_pspnet50_63cls $1
