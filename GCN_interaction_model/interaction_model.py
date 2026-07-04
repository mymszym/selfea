from model_train_test_func import *
from Param import *
import pickle


def main():
    fixed(SEED_NUM)
    print("----------------interaction model--------------------")
    cuda_num = CUDA_NUM
    print("GPU num {}".format(cuda_num))

    # Load train_ill and test_ill from ill_ent_ids (same as get_entity_embedding.py)
    ills = []
    with open(DATA_PATH + "ill_ent_ids", "r", encoding='utf-8') as f:
        for line in f:
            th = line.strip('\n').split('\t')
            ills.append((int(th[0]), int(th[1])))
    np.random.seed(2022)
    np.random.shuffle(ills)
    train_ill = ills[:int(len(ills) * 0.2)]  # 20% for training
    test_ill = ills[int(len(ills) * 0.2):]   # 80% for testing

    print("train_ill num: {} /test_ill num:{} / train_ill & test_ill num: {}".format(len(train_ill), len(test_ill), len(
        set(train_ill) & set(test_ill))))

    # (candidate) entity pairs
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))

    # interaction features
    nei_features = pickle.load(
        open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH, "rb"))  # neighbor-view interaction similarity feature
    att_features = pickle.load(
        open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH, 'rb'))  # attribute-view interaction similarity feature
    des_features = pickle.load(
        open(DESVIEW_SIMILARITY_FEATURE_PATH, "rb"))  # description/name-view interaction similarity feature
    train_candidate = pickle.load(open(TRAIN_CANDIDATES_PATH, "rb"))

    test_candidate = pickle.load(open(TEST_CANDIDATES_PATH, "rb"))
    all_features = []  # [nei-view cat att-view cat des/name-view]

    entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
    Train_gene = Train_index_generator(train_ill, train_candidate, entpair2f_idx, neg_num=NEG_NUM,
                                       batch_size=BATCH_SIZE)

    if HIERARCHY:
        hierarchy_features = pickle.load(open(HIERARCHY_FEATURE_PATH, "rb"))  # hierarchy feature
        for i in range(len(entity_pairs)):
            all_features.append(nei_features[i]
                                + att_features[i]
                                + des_features[i]
                                + hierarchy_features[i]
                                )  # 42 concat 42 concat 1.
        Model = MlP(KERNEL_NUM * 2 * 2 + 1 + 1, 11).cuda(cuda_num)

    else:
        for i in range(len(entity_pairs)):
            all_features.append(nei_features[i] + att_features[i] + des_features[i])  # 42 concat 42 concat 1.
        Model = MlP(KERNEL_NUM * 2 * 2 + 1, 11).cuda(cuda_num)
    print("All features embedding shape: ", np.array(all_features).shape)

    Optimizer = optim.Adam(Model.parameters(), lr=LEARNING_RATE)
    Criterion = nn.MarginRankingLoss(margin=MARGIN, size_average=True)

    # train
    train(Model, Optimizer, Criterion, Train_gene, all_features, test_candidate, test_ill,
          entpair2f_idx, epoch_num=EPOCH_NUM, eval_num=10, cuda_num=cuda_num, test_topk=50)

    # save
    torch.save(Model, open(INTERACTION_MODEL_SAVE_PATH, "wb"))


if __name__ == '__main__':
    main()
