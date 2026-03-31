export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export epochs=200
export num_workers=8

#python main_ood_data_aug.py --name "Batch_cutmix_intra_alpha_0.5"  --da_method "cutmix_intra" --Cutmix_alpha 0.5 --Cutmix_Prob 0.9  --epochs 200 --num_workers 8

#2024年10月6日验证
#for Cutmix_alpha in 0.5 1.0 2.0
#do
#	for da_method in 'cutmix_intra' 'cutmix_inter'
#	do
#		python main_ood_data_aug.py --name "Batch_${da_method}_alpha${Cutmix_alpha}" \
#			--da_method "${da_method}" \
#			--epochs "${epochs}" \
#			--num_workers "${num_workers}" \
#			--Cutmix_alpha ${Cutmix_alpha}
#	done
#done


#待验证
#python main_ood_data_aug.py --name "Batch_cutmix_intra_alpha0.05"  --da_method "cutmix_intra" --Cutmix_alpha 0.05 --Cutmix_Prob 0.9  --epochs 200 --num_workers 8


python main_ood_data_aug.py --name "Batch_cutmix_intra_alpha0.1p0.8"  --da_method "cutmix_intra" --Cutmix_alpha 0.1 --Cutmix_Prob 0.8  --epochs 200 --num_workers 8
python main_ood_data_aug.py --name "Batch_cutmix_intra_alpha0.1p1.0"  --da_method "cutmix_intra" --Cutmix_alpha 0.1 --Cutmix_Prob 1.0  --epochs 200 --num_workers 8

