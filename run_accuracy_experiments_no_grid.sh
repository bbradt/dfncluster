DATASETS=("gauss" "fbirn" "ucla")                                                  
CLUSTERER=("kmeans" "dbscan" "gmm" "bgmm" "hierarchical")                              
TIME=0                  
eval "$(conda shell.bash hook)"
conda activate dfncluster
for dataset in "${DATASETS[@]}"; do                                                    
    for clusterer in "${CLUSTERER[@]}"; do                                                 
        if [[ $dataset == "gauss" ]]; then                                             
            TIME=1                                                                         
	else                                                                               
	    TIME=0                                                                         
        fi                                                                                 
	python main.py --dataset $dataset --clusterer $clusterer --time_index $TIME
	python main.py --dataset $dataset --clusterer $clusterer --time_index $TIME --betas
    done                                                                                   
done                                                                                   
