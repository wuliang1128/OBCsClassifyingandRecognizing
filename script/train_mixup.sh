export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

# Net 1
#python src/train_pytorch.py \
#	--lr 0.1 \
#	--epochs 15 \
#	--net Net1 \
#	--data-dir ./data/oracle/

#python main_ood.py --name "train_layer_norm"

#python main.py --name "train_plain"


#断剑wl@WeChat@WeChat contact 2024 8-5 10:25:35AM

export epochs=200


export num_workers=8
#for alpha in 0.1 1.0 2.0
for alpha in 0.5
do
	for da_method in 'mixup_intra' 'mixup_inter'
	do
		python main_ood_data_aug.py --name "Batch_${da_method}_alpha${alpha}" \
			--da_method "${da_method}" \
			--epochs "${epochs}" \
			--num_workers "${num_workers}" \
			--alpha ${alpha}
	done
done



