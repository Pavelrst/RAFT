from os.path import exists


class raft_args():
    def __init__(self):
        self.experiment_name = 'default_args'
        self.dataset = None
        self.dataset_root = None
        self.num_of_workers = 0 #batch counter doesnt work for more than one worker
        self.restore_ckpt = None
        self.model = None
        self.small_model = False
        self.lr = 0.00002
        self.num_steps = 100000
        self.batch_size = 6
        self.iters = 12 # refinement iterations
        self.weight_decay = 0.00005
        self.epsilon = 1e-8
        self.clip = 1.0
        self.dropout = 0.0
        self.seed = 1234
        self.max_flow = 1000 # exclude extremly large displacements
        self.sum_freq = 10
        self.val_freq = 5000

        # Augmentation args
        self.augmentor_type = None

        # raft augmentor args
        self.image_size = [384, 512]
        self.min_scale = -0.2
        self.max_scale = 0.5

        # scopeflow augmentor args
        self.show_aug = False
        self.photometric_augmentations = False

        # tests before training
        self.test_dataset = False
        self.test_augmentor = True

        # validation indexes
        self.val_idx = [199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
                        340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358,
                        359, 360, 361, 362, 363, 364, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548,
                        549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 659, 660, 661, 662, 663, 664, 665,
                        666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684,
                        685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 967, 968, 969, 970, 971, 972,
                        973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991]

    def train_check_paths(self):
        assert exists(self.restore_ckpt), f"No such restore ckpt path {self.restore_ckpt}"
        assert exists(self.dataset_root), f"No such dataset path {self.dataset_root}"

    def val_check_paths(self):
        assert exists(self.dataset_root), f"No such dataset path {self.dataset_root}"
        assert exists(self.model), f"No such model path {self.model}"

# class raft_chairs_train_args(raft_args):
#     def __init__(self):
#         super().__init__()
#         self.experiment_name = 'chairs'
#         self.image_size = [368, 496]
#         self.dataset = 'chairs'
#         self.dataset_root = None
#         self.num_steps = 100000
#         self.lr = 0.0002
#         self.batch_size = 6
#         self.train_check_paths()

# class raft_things_train_args(raft_args):
#     def __init__(self):
#         super().__init__()
#         self.experiment_name = 'things'
#         self.image_size = [368, 768]
#         self.dataset = 'things'
#         self.dataset_root = None
#         self.num_steps = 60000
#         self.lr = 0.00005
#         self.batch_size = 3
#         self.restore_ckpt = 'checkpoints\\chairs.pth'
#         self.train_check_paths()

class raft_sintel_args(raft_args):
    def __init__(self):
        super().__init__()
        self.image_size = [368, 768] #'random_v3_raft' #'random_v2'
        self.padding = True
        self.dataset = 'sintel'
        # self.dataset_root = 'I:\\datasets\\Sintel\\training'
        self.dataset_root = 'datasets/Sintel/training' #'C:\\Users\\Pavel\\Downloads\\Sintel\\training'
        self.num_steps = 240000
        self.lr = 0.00005
        self.batch_size = 2
        self.restore_ckpt = 'checkpoints/chairs+things.pth' #'checkpoints\\chairs+things.pth'
        self.train_check_paths()
        # batch counter fix for scopflow random crop
        self.batch_crop_counter = crop_counter(self.batch_size)

class raft_sintel_ft_args(raft_sintel_args):
    def __init__(self, exp_name='sintel_ft_raft_aug', augment_type='raft_augmentor'):
        super().__init__()
        self.experiment_name = exp_name

        assert augment_type == 'scopeflow_augmentor' or augment_type == 'raft_augmentor'
        self.augmentor_type = augment_type


class raft_sintel_val_args(raft_args):
    def __init__(self, model):
        super().__init__()
        self.dataset = 'sintel'
        self.dataset_root = 'datasets/Sintel/training' #'I:\\datasets\\Sintel\\training'
        self.model = model
        self.iters = 50


class raft_sintel_debug_args(raft_args):
    def __init__(self):
        super().__init__()
        self.experiment_name = 'sintel_debug'
        self.dataset = 'sintel'
        self.dataset_root = 'I:\\datasets\\Sintel-debug\\training'
        self.num_steps = 100
        self.lr = 0.00005
        self.restore_ckpt = 'checkpoints\\chairs+things.pth'
        self.train_check_paths()
        self.augmentor_type = 'scopeflow_augmentor'
        self.show_aug = True

# fast fix.
import random
class crop_counter():
    def __init__(self, cycle_size):
        self.cycle_size = cycle_size
        self.curr_count = 0
        self.last_crop_size = None

    def to_string(self):
        return "count: "+str(self.curr_count)+" size: "+str(self.cycle_size)+" new_cycle:"+str(self.is_new_cycle())

    def inc(self):
        self.curr_count += 1
        if self.curr_count == self.cycle_size:
            self.curr_count = 0

    def get_crop_size(self):
        if self.is_new_cycle():
            crop_height, crop_width = random.choice([
                (432, 1024), (432, 1016), (432, 1008), (432, 992), (432, 984), (432, 976),
                (424, 1024), (424, 1016), (424, 1008), (424, 992), (424, 984), (424, 976),
                (416, 1024), (416, 1016), (416, 1008), (416, 992), (416, 984), (416, 976)])
            self.last_crop_size = crop_height, crop_width
            return crop_height, crop_width
        else:
            return self.last_crop_size

    def is_new_cycle(self):
        if self.curr_count == 0:
            return True
        else:
            return False