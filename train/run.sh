#python finetune.py -resume_skip_base  > ./../logs/log.txt

#python lwm_lwf_grad_cam.py -resume_skip_base > ./../logs/log_lwm.txt

#python lwf.py -resume_skip_base > ./../logs/log_lwf.txt

#ID=12181442
#python icarl_gradcam_loss.py -gpu_info "cuda:2" -resume_skip_base > ./../logs/${ID}_log_icarl_gradcam_loss.txt
python icarl.py  -resume_skip_base -base_num_classes 50 -incre_num_classes 5 -gpu_info "cuda:2" > ./../logs/log_icarl_160.txt

#python lucir.py -gpu_info "cuda:1"  -resume_skip_base -K 1 -base_num_classes 50 -incre_num_classes 1 > ./../logs/log_lucir.txt
#python podnet.py -gpu_info "cuda:1" -resume_skip_base > ./../logs/log_podnet.txt

#python ours.py  -gpu_info "cuda:2"  -resume_skip_base  > ./../logs/log_ours_3d.txt

#python ours_senet.py > ./../logs/log_ours_senet.txt

#python pod_improve.py -resume_skip_base > ./../logs/pod_improve.txt

