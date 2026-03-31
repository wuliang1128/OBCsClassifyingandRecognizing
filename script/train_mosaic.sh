export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export epochs=200
export num_workers=8

#训练效果不佳，放弃了

#python main_ood_data_aug.py --name "Batch_Mosaic_inter_p0.5"  --da_method "mosaic_inter" --Mosaic_Prob 0.5  --epochs 200 --num_workers 8

#2024年10月8日验证
for Mosaic_Prob in  0.05
do
	for da_method in 'mosaic_intra' 'mosaic_inter'
	do
		python main_ood_data_aug.py --name "Batch_${da_method}_p${Mosaic_Prob}" \
			--da_method "${da_method}" \
			--epochs "${epochs}" \
			--num_workers "${num_workers}" \
			--Mosaic_Prob ${Mosaic_Prob}
	done
done

