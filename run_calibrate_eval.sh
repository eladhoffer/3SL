checkpoints_dir="/home/ehoffer/PyTorch/3SL/results/mlperf_resnet50_bf16_baseline_all/checkpoints"
output_dir="mlperf_resnet50_bf16_cailbrated"
experiment="imagenet/mlperf_resnet50"
for EPOCH in 0{0..9} {10..38}
do
    python run.py experiment=${experiment} +evaluate="${checkpoints_dir}/epoch\=${EPOCH}.ckpt" name="${output_dir}/epoch_${EPOCH}" callbacks=calibrate_bn
done

CSV_FILES=''
for EPOCH in 0{0..9} {10..38}
do
    CSV_FILES="${CSV_FILES} ./results/${output_dir}/epoch_${EPOCH}/csv/version_0/metrics.csv"
done
csvstack $CSV_FILES > ./results/${output_dir}/metrics.csv