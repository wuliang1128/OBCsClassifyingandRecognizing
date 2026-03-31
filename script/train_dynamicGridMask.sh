export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export epochs=200
export num_workers=8


#python main_ood_data_aug.py --name "Batch_Gridmask"  --da_method "gridmask" --epochs 200 --num_workers 8
#python main_ood_data_aug.py --name "Batch_DynamicGridmask"  --da_method "dynamic" --epochs 200 --num_workers 8
#python main_ood_data_aug.py --name "Batch_DynamicGridmask_WithourR0.9k0.64"  --da_method "dynamic" --DynamicGridMask_DelProb 0.9 --epochs 200 --num_workers 8
#python main_ood_data_aug.py --name "Batch_DynamicGridmask_WithourR0.8k0.64"  --da_method "dynamic" --DynamicGridMask_DelProb 0.8  --epochs 200 --num_workers 8


#python main_ood_data_aug.py --name "Batch_DynamicGridmask_WithourR0.9k0.51d48_112"  --da_method "dynamic" --DynamicGridMask_DelProb 0.9  --epochs 200 --num_workers 8
#python main_ood_data_aug.py --name "Batch_DynamicGridmask_WithourR1.0k0.51d96_224"  --da_method "dynamic" --DynamicGridMask_DelProb 1.0  --epochs 200 --num_workers 8
#python main_ood_data_aug.py --name "Batch_DynamicGridmask_WithourR0.9k0.51d120_224"  --da_method "dynamic" --DynamicGridMask_DelProb 0.9  --epochs 200 --num_workers 8
#python main_ood_data_aug.py --name "Batch_DynamicGridmask_WithourR0.8k0.51d96_224"  --da_method "dynamic" --DynamicGridMask_DelProb 0.8  --epochs 200 --num_workers 8


for aspect_ratio in '1:1' '1:2'
do
  echo "Done processing."
  python main_ood_data_aug.py --name "Batch_DynamicGridmask_WithourR0.9k0.51d96_224_${aspect_ratio}" \
    --aspect_ratio "${aspect_ratio}" \
    --epochs "${epochs}" \
    --num_workers "${num_workers}"
done
