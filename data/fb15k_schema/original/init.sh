cat freebase_mtr100_mte100-*.txt | awk '{ print $1 "\n" $3 }' | sort | uniq > fb15k_entities.txt
cat freebase_mtr100_mte100-*.txt | awk '{ print $2 }' | sort | uniq > fb15k_relations.txt
