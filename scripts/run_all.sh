for patchSize in 3 5 7
do 
    for ((i=1;i<=10;i+=1))
    do
        filterSigma=$(bc <<<"scale=2; $i / 100" ) 
        for ((j=5;j<=15;j+=1))
        do
            patchSigma=$(bc <<<"scale=1; $j / 10" )
            ./build/main $patchSize $filterSigma $patchSigma
        done 
    done 
done