import os
import interface as bbox
import numpy as np
import random
import bisect
import sys
import unn
from unn import *
import threading
import subprocess
import os
import time
import traceback
import tempfile
import shutil
import cloudpickle
import pickle
import argparse


class SaneThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.finished = False
        self.func = func
        self.args = args
        self.terminate = False

    def run(self):
        try:
            self.subrun()
            self.finished = True
        except Exception:
            traceback.print_exc()

    def subrun(self):
        raise NotImplementedError()


class ThreadPool(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers

    def run(self, tasks, min2complete=None, max_fails=None):
        if max_fails is None:
            if min2complete is not None:
                max_fails = len(tasks) - min2complete
            else:
                max_fails = 0
        tasks = reversed(tasks)
        tasks = list(tasks)
        pool = []
        num_finished_tasks = 0
        num_tasks = len(tasks)
        while not (len(pool) == 0 and len(tasks) == 0):
            active_tasks = []
            for worker in pool:
                if worker.is_alive():
                    active_tasks.append(worker)
                elif not worker.finished:
                    if max_fails == 0:
                        assert False
                    max_fails -= 1
                else:
                    num_finished_tasks += 1

            pool = active_tasks
            if min2complete is None or num_finished_tasks < (min2complete + num_tasks) / 2:
                for i in reversed(range(len(tasks))):
                    if len(pool) == self.num_workers:
                        break
                    worker = tasks[i]
                    del tasks[i]
                    worker.daemon = True
                    worker.start()
                    pool.append(worker)
            if min2complete is not None:
                if num_finished_tasks > min2complete:
                    for worker in pool:
                        worker.terminate = True
                    return
            time.sleep(1)


class unique_tempdir(object):
    def __enter__(self):
        try:
            os.makedirs("/var/tmp/bb")
        except Exception:
            pass
        self.dir = tempfile.mkdtemp(dir="/var/tmp/bb")
        return self.dir
    def __exit__(self, etype, value, traceback):
        shutil.rmtree(self.dir)


class TaskWrapper(SaneThread):
    def __init__(self, func, args, result_file=None):
        SaneThread.__init__(self, func, args)
        self.result_file=result_file

    def subrun(self):
        with unique_tempdir() as tmp_f:
            with open(os.path.join(tmp_f, "input"), "w") as f:
                cloudpickle.dump((self.func, self.args), f)
            server = '''
import sys
import cloudpickle
import StringIO
input_ = StringIO.StringIO(sys.stdin.read())
stdout = sys.stdout
sys.stdout = StringIO.StringIO()
input_.seek(0)
(func, args) = cloudpickle.load(input_);
res = func(args)
for line in res:
    stdout.write(line)
                '''

            cmd = '''cd {cwd} && cat {input} | OMP_NUM_THREADS=1 nice -n +19 python -c "{server}" > {output} '''.format(
                    server=server,
                    input=os.path.join(tmp_f, "input"),
                    output=os.path.join(tmp_f, "output"),
                    cwd=os.path.dirname(os.path.abspath(__file__))
            )
            process = subprocess.Popen("/bin/bash -c '{}'".format(cmd.replace("'", "'\\''")),
                                       shell=True)
            while process.poll() is None:
                if self.terminate:
                    process.kill()
                time.sleep(1)
            nothing, err = process.communicate()
            retcode = process.poll()
            if retcode and not self.terminate:
                raise ProcessError(retcode, None, err)
            if self.result_file is not None:
                shutil.copy(os.path.join(tmp_f, "output"), self.result_file)

#-----------------------------------------------------------------


class LinearBot(object):
    def __init__(self, coeffs_path):
        n_features = 36
        n_actions = 4
        coeffs = np.loadtxt(coeffs_path).reshape(n_actions, n_features + 1)
        self.reg_coeffs = coeffs[:, :-1].T
        self.free_coeffs = coeffs[:, -1].T

    def reload(self):
        pass

    def get_values(self, state):
        return np.dot(state[:36], self.reg_coeffs) + self.free_coeffs

    def get_action(self, state):
        return np.argmax(self.get_values(state))


class UnnBot(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.apply_ = None
        self.reload()

    def reload(self):
        model = unn.load_model(self.model_path)
        computer = model.get_computer("output")
        for input_ in computer.get_inputs():
            if input_.name == "features":
                self.num_features = input_.num_features
        self.apply_ = unn.get_applier(computer)
    def get_values(self, state):
        features = state[:self.num_features]
        return self.apply_({"features": np.reshape(features, (1, len(features))).astype("float32")})[0]
    def get_action(self, state):
        return np.argmax(self.get_values(state))

class FasterUnnBot(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.reload()

    def reload(self):
        model = unn.load_model(self.model_path)
        computer = model.get_computer("output")
        for input_ in computer.get_inputs():
            if input_.name == "features":
                self.num_features = input_.num_features
        self.modules, inputs, output_id = computer.dump_fprop()

    def apply_(self, input_):
        output = input_
        for module in self.modules:
            func_name = module[0]
            params = module[1]
            if func_name == "standard_scaler":
                means, stds = params
                output = (output - means) / stds
            elif func_name == "rlu":
                output *= output > 0
            elif func_name == "dropout":
                output *= (1 - params[0])
            elif func_name == "affine":
                W, b = params
                output = output.dot(W) + b
            else:
                sys.stderr.write("unknown module: " + func_name)
                assert False
        return output
    def get_values(self, state):
        state = state[:self.num_features]
        return self.apply_(np.reshape(state, (1, len(state))).astype("float32"))
    def get_action(self, state):
        return np.argmax(self.get_values(state))

class Multibot(object):
    def __init__(self, bots, weights):
        self.bots = bots
        self.weights = weights
    def reload(self):
        for bot in self.bots:
            bot.reload()
    def get_action(self, state):
        res = self.weights[0] * self.bots[0].get_values(state)
        for bot, weight in zip(self.bots[1:], self.weights[1:]):
            res += bot.get_values(state) * weight
        return np.argmax(res)


class BBox(object):
    def __init__(self, level):
        self.level = level
        self.load_level()
        self.has_next = True
        self.prev_state = [0] * 36
        self.prev_action = 0
        self.pp_action = 0
        self.ppp_action = 0
        self.pppp_action = 0
        self.ppppp_action = 0
        self.checkpoints_map = {}

    def load_level(self):
        bbox.load_level(self.level, verbose=0)

    def get_state(self):
        return bbox.get_state()

    def do_action(self, action):
        self.ppppp_action = self.pppp_action
        self.pppp_action = self.ppp_action
        self.ppp_action = self.pp_action
        self.pp_action = self.prev_action
        self.prev_action = action
        self.prev_state = self.get_state()
        self.has_next = bbox.do_action(action)

    def get_score(self):
        return bbox.get_score()

    def get_max_time(self):
        return bbox.get_max_time()

    def get_time(self):
        return bbox.get_time()

    def create_checkpoint(self):
        id_ = bbox.create_checkpoint()
        self.checkpoints_map[id_] = (self.prev_action, self.pp_action, self.ppp_action, self.pppp_action, self.ppppp_action, self.prev_state)
        return id_

    def load_checkpoint(self, c):
        self.prev_action, self.pp_action, self.ppp_action, self.pppp_action, self.ppppp_action, self.prev_state = self.checkpoints_map[c]
        return bbox.load_from_checkpoint(c)

    def finish(self, verbose):
        bbox.finish(verbose=verbose)

    def reset_level(self):
        self.load_level()


class Strategy(object):
    def __init__(self, weight):
        self.weight = weight

    def get_name(self):
        assert False

class AsIsStrategy(Strategy):
    def __init__(self, weight):
        Strategy.__init__(self, weight)

    def apply(self, env, make_features):
        pass

    def get_name(self):
        return "asis"

class EpsGreedyStrategy(Strategy):
    def __init__(self, weight, bot, eps):
        Strategy.__init__(self, weight)
        self.eps = eps
        self.bot = bot

    def apply(self, env, make_features):
        num_steps = min(100 + random.expovariate(1.0 / 100), 300)
        for i in range(max(1, int(num_steps))):
            if random.random() < self.eps:
                env.do_action(random.randrange(4))
            else:
                env.do_action(self.bot.get_action(make_features(env)))

    def get_name(self):
        return "eps_greedy"


class RandomStrategy(Strategy):
    def __init__(self, weight):
        Strategy.__init__(self, weight)

    def apply(self, env, make_features):
        rand_val = random.random()
        if rand_val < 0.33:
            num_random_steps = min(random.expovariate(1.0 / 10), 300)
        elif rand_val < 0.67:
            num_random_steps = min(random.expovariate(1.0 / 30), 300)
        else:
            num_random_steps = min(random.expovariate(1.0 / 100), 300)
        for i in range(max(1, int(num_random_steps))):
            env.do_action(random.randrange(4))

    def get_name(self):
        return "random"

class RandomStartStrategy(Strategy):
    def __init__(self, bot, weight):
        Strategy.__init__(self, weight)
        self.bot = bot

    def apply(self, env, make_features):
        rand_val = random.random()
        if rand_val < 0.33:
            num_random_steps = min(random.expovariate(1.0 / 10), 300)
        elif rand_val < 0.67:
            num_random_steps = min(random.expovariate(1.0 / 30), 300)
        else:
            num_random_steps = min(random.expovariate(1.0 / 100), 300)
        for i in range(max(1, int(num_random_steps))):
            env.do_action(random.randrange(4))
        rand_val = random.random()
        if rand_val < 0.33:
            num_bot_steps = min(random.expovariate(1.0 / 10), 200)
        elif rand_val < 0.67:
            num_bot_steps = min(random.expovariate(1.0 / 30), 200)
        else:
            num_bot_steps = min(random.expovariate(1.0 / 100), 200)
        for i in range(max(1, int(num_bot_steps))):
            env.do_action(self.bot.get_action(make_features(env)))

    def get_name(self):
        return "random_start"

class MildRandomStrategy(Strategy):
    def __init__(self, weight):
        Strategy.__init__(self, weight)

    def apply(self, env, make_features):
        num_random_steps = min(random.expovariate(1.0 / 3), 300)
        for i in range(max(1, int(round(num_random_steps)))):
            env.do_action(random.randrange(4))

    def get_name(self):
        return "mild_random"


class MildRandomStartStrategy(Strategy):
    def __init__(self, bot, weight):
        Strategy.__init__(self, weight)
        self.bot = bot

    def apply(self, env, make_features):
        num_random_steps = min(random.expovariate(1.0 / 5), 300)
        for i in range(max(1, int(round(num_random_steps)))):
            env.do_action(random.randrange(4))
        num_bot_steps = min(random.expovariate(1.0 / 5), 200)
        for i in range(max(1, int(round(num_bot_steps)))):
            env.do_action(self.bot.get_action(make_features(env)))

    def get_name(self):
        return "mild_random_start"


class MixedRandomStrategy(Strategy):
    def __init__(self, bot, steps, scaler, weight):
        Strategy.__init__(self, weight)
        self.bot = bot
        self.steps = steps
        self.scaler = scaler

    def apply(self, env, make_features):
        start_time = env.get_time()
        num_steps = min(random.expovariate(1.0 / self.steps), 500)
        for cnt in range(1000):
            if cnt % 2 == 0:
                num_random_steps = max(1, min(random.expovariate(1.0 / 3 / self.scaler), 10 * self.scaler))
                num_steps -= num_random_steps
                if num_steps < 0:
                    break
                for i in range(int(num_random_steps)):
                    env.do_action(random.randrange(4))
            else:
                num_bot_steps = max(1, min(random.expovariate(1.0 / 10 / self.scaler), 30 * self.scaler))
                num_steps -= num_bot_steps
                if num_steps < 0:
                    break
                for i in range(int(num_bot_steps)):
                    env.do_action(self.bot.get_action(make_features(env)))
        assert env.get_time() - start_time <= 500

    def get_name(self):
        return "mixed_random"


class Strategies(object):
    def __init__(self, strategies):
        self.strategies = strategies
        self.weights = np.cumsum([s.weight for s in strategies])

    def get(self):
        pos = bisect.bisect_left(self.weights, self.weights[-1] * random.random())
        return self.strategies[pos]


def get_actions(depth):
    all_actions = []
    num_actions = int(pow(4, depth))
    for action_idx in range(num_actions):
        all_actions.append([])
        cur_idx = action_idx
        for cur_depth in range(depth):
            all_actions[-1].append(cur_idx % 4)
            cur_idx /= 4
    return all_actions


def score_actions(env, make_features, bot, tail, depth=1):
    checkpoint = env.create_checkpoint()
    start_score = env.get_score()
    scores = [-1e9, -1e9, -1e9, -1e9]

    all_actions = get_actions(depth)
    for actions in all_actions:
        env.load_checkpoint(checkpoint)
        for action in actions:
            env.do_action(action)
        for _ in range(tail):
            env.do_action(bot.get_action(make_features(env)))
        final_score = env.get_score()
        scores[actions[0]] = max(scores[actions[0]], final_score - start_score)
    env.load_checkpoint(checkpoint)
    return scores

def get_features1(env):
    return env.get_state()

def get_features3(env):
    return list(env.get_state()) + [int(env.prev_action == 0),
                int(env.prev_action == 1), int(env.prev_action == 2), int(env.prev_action ==3)]

def build_train_job(args):
    idx, start_bot, bot, strategies, interval, tail, level, make_features = args
    random.seed(str(idx))
    env = BBox(level)
    env.do_action(start_bot.get_action(make_features(env)))
    res = []
    while env.get_max_time() - env.get_time() > tail + 20:
        start_time = env.get_time()
        strategy = strategies.get()
        strategy.apply(env, make_features)
        features = make_features(env)
        if env.get_time() - start_time > 500:
            sys.stderr.write("Strategy {} took more steps than allowed: {}->{}\n".format(
                type(strategy), start_time, env.get_time()))
            assert env.get_time() - start_time <= 500
        scores = score_actions(env, make_features, bot, tail, depth=1)
        if any(item != scores[0] for item in scores):
            res.append(" ".join(str(item) for item in scores) + "\t" +
                          " ".join(str(item) for item in features) +
                          "\t" + strategy.get_name() + "\n")

        for i in range(int(interval * random.random())):
            if env.has_next:
                env.do_action(start_bot.get_action(make_features(env)))
    return res


def build_train(start_bot, bot, strategies, interval, tail, levels, work_dir, num_tasks, feature_maker, num_threads):
    try:
        shutil.rmtree(os.path.join(work_dir, "train/chunks"))
    except Exception:
        pass
    try:
        os.makedirs(os.path.join(work_dir, "model"))
    except Exception:
        pass
    os.makedirs(os.path.join(work_dir, "train/chunks"))
    tasks = []
    for i in range(num_tasks):
        tasks.append(TaskWrapper(
            func=build_train_job,
            args=(i, start_bot, bot, strategies, interval, tail, levels[i % len(levels)], feature_maker),
            result_file=os.path.join(work_dir, "train/chunks/features.tsv.") + str(i)))
    pool = ThreadPool(num_threads)
    pool.run(tasks, min2complete=int(len(tasks) * 0.9), max_fails=(0.1 * len(tasks)))
    subprocess.check_call("cat {chunks_dir}/* | shuf | cut -f 1,2 > {features_file}".format(
        chunks_dir=os.path.join(work_dir, "train/chunks"),
        features_file=os.path.join(work_dir, "model/train.tsv")
    ), shell=True)


def build_test_job(args):
    try:
        time, bot, tail, env, make_features = args
        env.reset_level()
        while env.get_time() < time:
            env.do_action(bot.get_action(make_features(env)))
        start_time = env.get_time()
        res = []
        while env.get_max_time() - env.get_time() > tail + 20 and env.get_time() - start_time <= 20000:
            scores = score_actions(env, make_features, bot, tail, depth=1)
            if any(item != scores[0] for item in scores):
                res.append(" ".join(str(item) for item in scores) + "\t" +
                             " ".join(str(item) for item in make_features(env)) + "\n")
            env.do_action(bot.get_action(make_features(env)))
            while random.random() < 0.5:
                env.do_action(bot.get_action(make_features(env)))
        return res
    except Exception as ex:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print 'An error occurred on line {} in statement {}'.format(line, text)
        raise


def build_test(bot, tail, level, work_dir, feature_maker, num_threads):
    try:
        shutil.rmtree(os.path.join(work_dir, "test/chunks"))
    except Exception:
        pass
    os.makedirs(os.path.join(work_dir, "test/chunks"))
    try:
        os.makedirs(os.path.join(work_dir, "model"))
    except Exception:
        pass
    times = []
    env = BBox(level)
    cur_time = 0
    while env.get_max_time() - cur_time > tail + 20:
        if cur_time % 20000 == 1:
            times.append(cur_time)
        cur_time += 1
    tasks = []
    for time in times:
        tasks.append(TaskWrapper(
            func=build_test_job,
            args=(time, bot, tail, env, feature_maker),
            result_file=os.path.join(work_dir, "test/chunks/features.tsv.") + str(time)))
    pool = ThreadPool(num_threads)
    pool.run(tasks, max_fails=(0.1 * len(tasks)))
    subprocess.check_call("cat {chunks_dir}/* | shuf | cut -f 1,2 > {features_file}".format(
            chunks_dir=os.path.join(work_dir, "test/chunks"),
            features_file=os.path.join(work_dir, "model/test.tsv")
        ),
        shell=True
    )

def test_bot(bot, level, make_features):
    env = BBox(level)
    while env.has_next:
        if env.get_time() % 10000 == 0:
            print str(env.get_time()) + "\t" + str(env.get_score())
        action = bot.get_action(make_features(env))
        env.do_action(action)
    bbox.finish()
    print bbox.get_score()

def load_bot(path):
    if os.path.isfile(os.path.join(path, "model.unn")):
        bot = FasterUnnBot(os.path.join(path, "model.unn"))
    else:
        bot = LinearBot(os.path.join(path, "reg_coefs.txt"))
    return bot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["build", "test"])
    parser.add_argument("-t", "--threads", type=int, default=16, help="Number of threads")
    parser.add_argument("-s", "--strategies", type=str, choices=["s3", "s6"])
    parser.add_argument("--start_bot", help="Start bot folder", default=None)
    parser.add_argument("--bot", help="Bot folder", nargs="+")
    parser.add_argument("--tasks", help="Number of tasks to do during training", default=1000, type=int)
    parser.add_argument("--tail", help="Length of rollouts", default=100, type=int)
    parser.add_argument("--interval", help="Interval between strategies", default=200, type=int)
    parser.add_argument("-f", "--folder", type=str, help="folder where to put the results")
    parser.add_argument("-r", "--reverse", action="store_const", const=True, default=False, help="Reverse train and test")
    parser.add_argument("-a", "--all", action="store_const", const=True, default=False, help="Train on train and test")
    parser.add_argument("--level", help="Path to the test level", default=None)
    parser.add_argument("--features", choices=["set1", "set3"], default="set1")
    args = parser.parse_args()

    if args.features == "set1":
        feature_maker = get_features1
    elif args.features == "set3":
        feature_maker = get_features3
    else:
        assert False

    bots = []
    weights = []
    for bot in args.bot:
        if ":" in bot:
            weight, path = bot.split(":")
            bots.append(load_bot(path))
            weights.append(float(weight))
        else:
            bots.append(load_bot(bot))
            weights.append(1)
    if len(bots) == 1:
        bot = bots[0]
    else:
        bot = Multibot(bots, weights)

    if args.start_bot is not None:
        start_bot = load_bot(args.start_bot)
    else:
        start_bot = bot

    if args.strategies == "s3":
        strategies = Strategies([EpsGreedyStrategy(1, start_bot, 0.1), EpsGreedyStrategy(1, start_bot, 0.05),
                                 RandomStrategy(1), RandomStartStrategy(start_bot, 1),
                                 MildRandomStrategy(1), MildRandomStartStrategy(start_bot, 1),
                                 MixedRandomStrategy(start_bot, 100, 1, 1), MixedRandomStrategy(start_bot, 30, 1, 1),
                                 MixedRandomStrategy(start_bot, 300, 1, 0.5), MixedRandomStrategy(start_bot, 300, 2, 0.5),
                                 MixedRandomStrategy(start_bot, 300, 4, 0.5)])
    elif args.strategies == "s6":
        strategies = Strategies([MildRandomStrategy(1)])
    else:
        assert args.mode == "test"

    if not args.reverse:
        train_levels = ["train_level.data"]
        test_level = "test_level.data"
    else:
        train_levels = ["test_level.data"]
        test_level = "train_level.data"

    if args.all:
        train_levels = ["train_level.data",
                        "test_level.data"]

    if args.mode == "build":
        print "building train"
        if not os.path.isdir(args.folder):
            os.makedirs(args.folder)
        build_train(start_bot, bot, tail=args.tail, strategies=strategies, levels=train_levels,
                    interval=args.interval,
                    work_dir=args.folder, num_tasks=args.tasks,
                    feature_maker=feature_maker, num_threads=args.threads)
        print "building test"
        build_test(bot, tail=args.tail, level=test_level,work_dir=args.folder,
                   feature_maker=feature_maker, num_threads=args.threads)
    elif args.mode =="test":
        if args.level is not None:
            test_level = args.level
        test_bot(bot, test_level, feature_maker)
    else:
        sys.stderr.write("Unknown mode: " + args.mode + "\n")

if __name__ == "__main__":
    main()
