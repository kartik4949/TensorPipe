ls tests/*.py|xargs -n 1 -P 3 python &> test.log &&echo "All passed" || echo "Failed! Search keyword FAILED in test.log"

