# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=5

###############################################################
# save directory
OUTDIR=outputs/CIFAR100-10k/realistic-rand
# VISUALISATIONDIR = visualisation/
MAXTASK=-1

# hard coded inputs
REPEAT=3
SCHEDULE="200"
MODELNAME="WideResNet_28_2_cifar"
MODELNAMEOOD_DC="WideResNet_DC_28_2_cifar"
L_DIST="super"
UL_DIST="rand"
BS=64
UBS=128
WD=5e-4

# GD parameters
SCHEDULE_GD="120 160 180 200"

# realistic specific parameters
MEMORY=400

# tuned paramaters
LR_PL=0.1
WA_PL=1
TPR=0.05
TPR_OOD=0.05
LR_GD=0.1
Co_GD=1.0

###############################################################

# process inputs
mkdir -p $OUTDIR
# mkdir -p $VISUALISATIONDIR

# dm - without threshold warmup
# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/no_warmup --dynamic_threshold True --fm_thresh 0.95 --fm_epsilon 0.000001

# dm - with threshold warmup
# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/warmup --dynamic_threshold False --fm_thresh 0.95 --fm_epsilon 0.000001 --threshold_warmup True

# # dm - with threshold warmup and non linear mapping
# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/warmup_non_linear --dynamic_threshold False --fm_thresh 0.95 --fm_epsilon 0.000001 --threshold_warmup True --non_linear_mapping True

##########################################
#       Check impact of losses           #
##########################################

# Without dynamic threshold

# Removed Distillation loss
# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/no_distillation/ --fm_thresh 0.85 --fm_epsilon 0.000001 --is_unsupervised_loss True 

# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/weighted_supervised/ --fm_thresh 0.85 --fm_epsilon 0.000001

# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/unweighted_supervised/ --fm_thresh 0.85 --fm_epsilon 0.000001 --is_unsupervised_loss True


##########################################
#           TSNE PLOTTING                #
##########################################

# python -u tsne_plot.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/no_warmup --dynamic_threshold True --fm_thresh 0.95 --fm_epsilon 0.000001 

# dm - with threshold warmup
# python -u tsne_plot.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/warmup --dynamic_threshold False --fm_thresh 0.95 --fm_epsilon 0.000001 --threshold_warmup True --visualisation_dir ${VISUALISATIONDIR}

# # dm - with threshold warmup and non linear mapping
# python -u tsne_plot.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
#     --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
#     --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
#     --weight_aux $WA_PL --fm_loss \
#     --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
#     --tpr $TPR --oodtpr $TPR_OOD \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/dm/warmup_non_linear --dynamic_threshold False --fm_thresh 0.95 --fm_epsilon 0.000001 --threshold_warmup True --non_linear_mapping True --visualisation_dir ${VISUALISATIONDIR}


###################################################################################################################################################################################################












# # end to end incremental learning
# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#         --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
#         --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
#         --memory $MEMORY --model_name $MODELNAME --model_type resnet --DW --FT \
#         --learner_type distillation --learner_name GD --co 0.0 --distill_loss L   \
#         --max_task $MAXTASK --log_dir ${OUTDIR}/ete

# # global distillation
# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#     --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
#     --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
#     --memory $MEMORY --model_name $MODELNAME --model_type resnet --DW --FT \
#     --learner_type distillation --learner_name GD --co $Co_GD --distill_loss P C Q    \
#     --max_task $MAXTASK --log_dir ${OUTDIR}/gd

# # distillation and retrospection
# python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
#         --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
#         --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
#         --memory $MEMORY --model_name $MODELNAME --model_type resnet  \
#         --learner_type distillation --learner_name GD --co 0.0 --distill_loss L C    \
#         --max_task $MAXTASK --log_dir ${OUTDIR}/dr

python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 --ul_dist $UL_DIST  \
    --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
    --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
    --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
    --weight_aux $WA_PL --fm_loss \
    --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --FT  \
    --tpr $TPR --oodtpr $TPR_OOD \
    --max_task $MAXTASK --log_dir ${OUTDIR}/dm/unweighted_undistilled/ --fm_thresh 0.85 --fm_epsilon 0.000001 --is_unsupervised_loss True