from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
import logging

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pickle
import time
from Param import *
from utils import fixed, cos_sim_mat_generate, batch_topk
from GCN_basic_bert_unit.Basic_Bert_Unit_model import Basic_Bert_Unit_model
from GCN_basic_bert_unit.GCN_mode import *
CUDA = 0
device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA < torch.cuda.device_count() else "cpu")

def candidate_generate(ents1, ents2, ent_emb, candidate_topk=50, bs=32, cuda_num=0):
    """
    return a dict, key = entity, value = candidates (likely to be aligned entities)
    """
    emb1 = np.array(ent_emb)[ents1].tolist()
    emb2 = np.array(ent_emb)[ents2].tolist()
    print("Test(get candidate) embedding shape:", np.array(emb1).shape, np.array(emb2).shape)
    print("get candidate by cosine similartity.")
    res_mat = cos_sim_mat_generate(emb1, emb2, bs, cuda_num=cuda_num)

    score, index = batch_topk(res_mat, bs, candidate_topk, largest=True, cuda_num=cuda_num)
    ent2candidates = dict()
    for i in range(len(index)):
        e1 = ents1[i]
        e2_list = np.array(ents2)[index[i]].tolist()
        ent2candidates[e1] = e2_list
    return ent2candidates


def all_entity_pairs_gene(candidate_dict_list, ill_pair_list):
    # generate list of all candidate entity pairs.
    entity_pairs_list = []
    for candidate_dict in candidate_dict_list:
        for e1 in candidate_dict.keys():
            for e2 in candidate_dict[e1]:
                entity_pairs_list.append((e1, e2))
    for ill_pair in ill_pair_list:
        for e1, e2 in ill_pair:
            entity_pairs_list.append((e1, e2))
    entity_pairs_list = list(set(entity_pairs_list))
    print("entity_pair (e1,e2) num is: {}".format(len(entity_pairs_list)))
    return entity_pairs_list


def main():
    fixed(SEED_NUM)
    print("----------------get entity embedding--------------------")
    cuda_num = CUDA_NUM
    batch_size = 256
    print("GPU NUM:", cuda_num)

    # read other data from bert unit model(eid2data, triples)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
    _, _, eid2data, triples = pickle.load(open(bert_model_other_data_path, "rb"))

    # Load train_ill and test_ill from ill_ent_ids (not from saved other_data.pkl)
    ills = []
    with open(DATA_PATH + "ill_ent_ids", "r", encoding='utf-8') as f:
        for line in f:
            th = line.strip('\n').split('\t')
            ills.append((int(th[0]), int(th[1])))
    np.random.seed(2022)
    np.random.shuffle(ills)
    train_ill = ills[:int(len(ills) * 0.2)]  # 20% for training
    test_ill = ills[int(len(ills) * 0.2):]   # 80% for testing

    print("train_ill num: {} /test_ill num: {} / train_ill & test_ill num: {}".format(len(train_ill), len(test_ill),
                                                                                      len(set(train_ill) & set(
                                                                                          test_ill))))

    # load basic bert unit model

    # bert_model_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + "model_epoch_" \
    #                   + str(LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM) + '.p'
    bert_model_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + "model" + '.p'
    Model = combine_model(len(eid2data), triples=triples)
    # 加载模型时忽略不匹配的键（比如 entity_emb 可能是随机初始化的）
    state_dict = torch.load(bert_model_path, map_location='cpu')
    missing_keys, unexpected_keys = Model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in state_dict (will use random init): {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict (ignored): {unexpected_keys}")
    print("loading basic bert unit model from:  {}".format(bert_model_path))
    Model.eval()
    for name, v in Model.named_parameters():
        v.requires_grad = False
    Model = Model.to(device)

    # generate entity embedding by basic bert unit
    start_time = time.time()
    ent_emb = []
    for eid in range(0, len(eid2data.keys()), batch_size):  # eid == [0,n)
        token_inputs = []
        mask_inputs = []
        for i in range(eid, min(eid + batch_size, len(eid2data.keys()))):
            token_input = eid2data[i][0]
            mask_input = eid2data[i][1]
            token_inputs.append(token_input)
            mask_inputs.append(mask_input)
        vec = Model(torch.LongTensor(token_inputs).to(device),
                    torch.FloatTensor(mask_inputs).to(device),
                    list(range(eid, min(eid + batch_size, len(eid2data.keys())))))
        ent_emb.extend(vec.detach().cpu().tolist())
    print("get entity embedding using time {:.3f}".format(time.time() - start_time))
    print("entity embedding shape: ", np.array(ent_emb).shape)

    # save entity embedding.
    pickle.dump(ent_emb, open(ENT_EMB_PATH, "wb"))
    print("save entity embedding....")

    # Generate candidates(likely to be aligned) for entities in train_set/test_set
    # we apply interaction model to infer a matching score on candidates.
    test_ids_1 = [e1 for e1, e2 in test_ill]
    test_ids_2 = [e2 for e1, e2 in test_ill]
    train_ids_1 = [e1 for e1, e2 in train_ill]
    train_ids_2 = [e2 for e1, e2 in train_ill]
    train_candidates = candidate_generate(train_ids_1, train_ids_2, ent_emb, CANDIDATE_NUM, bs=2048,
                                          cuda_num=CUDA_NUM)
    test_candidates = candidate_generate(test_ids_1, test_ids_2, ent_emb, CANDIDATE_NUM, bs=2048, cuda_num=CUDA_NUM)
    pickle.dump(train_candidates, open(TRAIN_CANDIDATES_PATH, "wb"))
    print("save candidates for training ILL data....")
    pickle.dump(test_candidates, open(TEST_CANDIDATES_PATH, "wb"))
    print("save candidates for testing ILL data....")

    # entity_pairs (entity_pairs is list of (likely to be aligned) entity pairs : [(e1,ea),(e1,eb),(e1,ec) ....])
    entity_pairs = all_entity_pairs_gene([train_candidates, test_candidates], [train_ill])
    pickle.dump(entity_pairs, open(ENT_PAIRS_PATH, "wb"))
    turelink = 0
    for (h, r) in entity_pairs:
        if (h, r) in test_ill:
            turelink += 1
    print("tureNum:{} testNum:{} rate:{}".format(turelink, len(test_ill), turelink / len(test_ill)))
    print("save entity_pairs save....")
    print("entity_pairs num: {}".format(len(entity_pairs)))


if __name__ == '__main__':
    main()
