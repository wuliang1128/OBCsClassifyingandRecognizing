export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export epochs=200
export num_workers=8

#训练效果不佳，放弃了
#python main_ood_data_aug.py --name "Batch_Mosaic_inter_p0.5"  --da_method "mosaic_inter" --Mosaic_Prob 0.5  --epochs 200 --num_workers 8


#for da_method in 'combination224_intra' 'combination333_intra' 'combination111_intra'
#for da_method in  'combination333_intra' 'combination224_intra'
#for da_method in  'combination224_intra' 'combination333_intra' 'combination12_intra' 'combination13_intra' 'combination23_intra' 'mixup_intra' 'dynamic' 'cutmix_intra' 'base'
for da_method in  'combination333_intra' 'combination12_intra' 'combination13_intra' 'combination23_intra' 'mixup_intra' 'dynamic' 'cutmix_intra' 'base'
do
  python main_ood_data_aug.py --name "Batch_${da_method}_new" \
    --da_method "${da_method}" \
		--epochs "${epochs}" \
	  --num_workers "${num_workers}"
done

