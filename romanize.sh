for i in $1; 
    do perl /home/aip000/sgile/code/uroman/bin/uroman.pl < $i > ${i%????}_romanized.txt; 
done;