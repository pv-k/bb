My solution to the blackbox challenge: http://blackboxchallenge.com

To train a single bot, that gets 3500-3700 on the leaderboard do the following:
1.  Download unn.py (https://github.com/pv-k/unn), make sure all necessary libraries are installed. Add it to PATH and PYTHONPATH.
2.  Create a folder {root}, copy train_level.data, test_level.data, interface.so, reg_coefs.txt, bot.py there. Don't place anything like *.unn in root.
3.  Building the dataset. Run the command. It will take hours. Progress can be monitored in s6/int200/v1a/train/chunks, number of chunks is the number of tasks. This command uses both levels (train and test) for training.
OMP_NUM_THREADS=1 python bot.py --mode build -t 16 -s s6 --bot . --tasks 1000 --tail 100 --interval 200 -f s6/int200/v1a -a --features set3
-t is the number of threads, the higher the better.  More --tasks should result in a better score.
4.  Training the model. Go to s6/int200/v1a/model. Run
unn.py learn -m model.unn -f train.tsv -t test.tsv --mt --mf -i scores:dense:4,features:dense:40@scale -a 'output=linear(dropout(rlu(features,100),0.3), 4)|energy=ranknet(output,scores)' --te energy --batch_size 30 -o nesterov --lr 0.005 --epochs 5
5.  Testing. Go to root. Run
OMP_NUM_THREADS=1 python bot.py --mode test --bot s6/int200/v1a/model 0.01:. --features set3 –-level {path_to_the_level}