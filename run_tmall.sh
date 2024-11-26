#python main.py \
#        --dataset Nowplaying
export CUDA_VISIBLE_DEVICES=0

NEI_N=(2 3 4 5 6 7)
GAMA=(2.3 2.0 1.7 1.4 1.1 0.8)
CIR=(1 2)


#for gm in ${GAMA[@]}
#  do
#      python main.py \
#        --dataset Nowplaying\
#        --gama $gm
#  done

for nei_n in ${NEI_N[@]}
  do
      python main.py \
        --dataset Tmall\
        --nei_n $nei_n
  done