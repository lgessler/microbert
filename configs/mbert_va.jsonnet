local embedding_dim = std.parseInt(std.extVar("embedding_dim"));
local bert_model = std.extVar("bert_model");

// --------------------------------------------------------------------------------
// Reader setup
// --------------------------------------------------------------------------------
local readers = std.parseJson(std.extVar("readers"));

// --------------------------------------------------------------------------------
// Data path setup
// --------------------------------------------------------------------------------
local train_data_paths = std.parseJson(std.extVar("train_data_paths"));
local dev_data_paths = std.parseJson(std.extVar("dev_data_paths"));

// --------------------------------------------------------------------------------
// Head setup
// --------------------------------------------------------------------------------
local mlm_head = {
  "type": "mlm",
  "embedding_dim": embedding_dim
};
local heads = {"mlm": mlm_head};
local weights = {"mlm": 1.0};

// scheduling
local batch_size = 8;
local instances_per_epoch = 32000;
local batches_per_epoch = instances_per_epoch / batch_size;
local num_epochs = 20; // BERT_base_total_instances / instances_per_epoch;

{
    "dataset_reader" : {
        "type": "multitask",
        "readers": readers
    },
    "data_loader": {
        "type": "multitask_ldg",
        "scheduler": {
            "type": "homogeneous_roundrobin",
            "batch_size": batch_size,
        },
        "shuffle": true,
        "instances_per_epoch": instances_per_epoch,
        "sampler": {
            "type": "weighted",
            "weights": weights
        }
    },
    "train_data_path": train_data_paths,
    "validation_data_path": dev_data_paths,
    "model": {
        "type": "multitask",
        "backbone": {
            "type": "pretrained_bert",
            "bert_model": bert_model
        },
        "heads": heads
    },
    "trainer": {
        "type": "mtl",
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.05
        },
        "num_epochs": 20,
        "validation_metric": "-mlm_perplexity",
        "callbacks": [
            {
                "type": "console_logger"
            },
            {
                "type": "tensorboard"
            }
        ]
    }
}
