import unittest
from classifier import *

EXPECTED_EVAL_ACCURACY = .48
EXPECTED_EVAL_LOSS =  0.6538470834493637
EXPECTED_GLOBALSTEP = 2
EXPECTED_LOSS = 0.6450820863246918 

class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.bert_model = "bert-base-uncased" #"./script_test_data"
        self.label_list = ["left", "right"]
        self.num_labels = NUM_LABELS
        self.tokenizer  = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        self.processor = LeftRightProcessor()
        self.output_dir = "./script_test_data"
        self.cache_dir = ""
        self.data_dir = "./data"
        self.local_rank = -1
        self.max_seq_length = 128
        self.no_cuda = False
        self.eval_batch_size = 8
        self.num_train_examples = 50
        self.num_test_examples = 25
        self.train_batch_size = 32
        self.gradient_accumulation_steps = 1
        self.num_train_epochs = 1
        self.learning_rate = 5e-5
        self.warmup_proportion = .1
        self.device, self.n_gpu = get_device_and_n_gpu(self.no_cuda, self.local_rank)
        self.step_tracker = StepValues()
        self.seed = 42
        seed_all(self.seed, self.n_gpu)
        self.model = get_model(
            self.cache_dir, 
            self.local_rank, 
            self.bert_model, 
            self.num_labels, 
            self.device, 
            self.n_gpu
        )

    def test_train_and_eval(self):
        train_examples = self.processor.get_train_examples(self.data_dir, self.num_train_examples, self.seed)
        num_train_optimization_steps = int(len(train_examples) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_train_epochs
        if self.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        optimizer = get_optimizer(self.model, self.learning_rate, self.warmup_proportion, num_train_optimization_steps)

        train_dataloader = get_train_dataloader(
            train_examples, 
            self.tokenizer, 
            self.label_list, 
            self.train_batch_size, 
            self.max_seq_length, 
            num_train_optimization_steps, 
            self.local_rank
        ) 
        self.step_tracker = train_model(
            self.model,
            optimizer, 
            train_dataloader, 
            self.num_train_epochs, 
            self.gradient_accumulation_steps, 
            self.device, 
            self.n_gpu
        )

        eval_dataloader = get_eval_dataloader(
            self.processor, 
            self.data_dir,
            self.num_test_examples, 
            self.eval_batch_size, 
            self.max_seq_length, 
            self.tokenizer,
            self.seed
        )
        result = eval_model(self.model, self.device, eval_dataloader, self.step_tracker)

        expected_result = {
            "eval_accuracy":EXPECTED_EVAL_ACCURACY, 
            "eval_loss":EXPECTED_EVAL_LOSS, 
            "global_step":EXPECTED_GLOBALSTEP, 
            "loss":EXPECTED_LOSS
        }
        self.assertAlmostEqual(result["eval_accuracy"], expected_result["eval_accuracy"], 5)
        self.assertAlmostEqual(result["eval_loss"], expected_result["eval_loss"], 5)
        self.assertAlmostEqual(result["global_step"], expected_result["global_step"], 5)
        self.assertAlmostEqual(result["loss"], expected_result["loss"], 1)





if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(IntegrationTest('test_train_and_eval'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
