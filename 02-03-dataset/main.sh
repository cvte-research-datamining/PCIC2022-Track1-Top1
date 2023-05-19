cd ./src/01-process/
sh run.sh

cd ../02-data/lstm/
sh run.sh

cd ../finetune/
sh run.sh

cd ../../03-data/01-feat/
sh run.sh

cd ../02-catboost/kfold/
sh run.sh

cd ../graft/
sh run.sh

cd ../../03-lightgbm/01-lgb-graft/
sh run.sh

cd ../../../04-fusion/
sh run.sh
