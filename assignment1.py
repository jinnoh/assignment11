import gzip
import ast
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import random
from collections import defaultdict

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class MFReg(nn.Module):
    def __init__(self, n_u, n_b, k):
        super().__init__()
        self.u_vec = nn.Embedding(n_u, k)
        self.b_vec = nn.Embedding(n_b, k)
        self.u_b = nn.Embedding(n_u, 1)
        self.b_b = nn.Embedding(n_b, 1)
        self.mu = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.u_vec.weight, std=0.01)
        nn.init.normal_(self.b_vec.weight, std=0.01)
        nn.init.zeros_(self.u_b.weight)
        nn.init.zeros_(self.b_b.weight)

    def forward(self, u, b):
        p = self.u_vec(u)
        q = self.b_vec(b)
        s = (p * q).sum(dim=1, keepdim=True)
        y = self.mu + self.u_b(u) + self.b_b(b) + s
        return y.squeeze(1)

    def reg(self):
        return (
            self.u_vec.weight.pow(2).sum()
            + self.b_vec.weight.pow(2).sum()
            + self.u_b.weight.pow(2).sum()
            + self.b_b.weight.pow(2).sum()
        )


def make_maps(df):
    us = sorted(df["userID"].unique())
    bs = sorted(df["bookID"].unique())
    um = {u: i for i, u in enumerate(us)}
    bm = {b: i for i, b in enumerate(bs)}
    return um, bm


def df_to_arrays(df, um, bm, col):
    u_map = df["userID"].map(um)
    b_map = df["bookID"].map(bm)
    m = u_map.notna() & b_map.notna()
    u = u_map[m].astype(np.int64).to_numpy()
    b = b_map[m].astype(np.int64).to_numpy()
    y = df.loc[m, col].astype(np.float32).to_numpy()
    return u, b, y


def train_rating_mf(df, k, epochs, batch, lr, wd, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    um, bm = make_maps(df)
    n_u = len(um)
    n_b = len(bm)
    u_idx, b_idx, r = df_to_arrays(df, um, bm, "rating")
    u_t = torch.from_numpy(u_idx)
    b_t = torch.from_numpy(b_idx)
    r_t = torch.from_numpy(r)
    n = len(u_t)
    model = MFReg(n_u, n_b, k)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in tqdm(range(0, n, batch), desc=f"rating_epoch_{ep+1}", leave=False):
            idx = perm[s:s+batch]
            uu = u_t[idx]
            bb = b_t[idx]
            rr = r_t[idx]
            pred = model(uu, bb)
            loss = mse(pred, rr) + wd * model.reg()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, um, bm


def predict_rating_single(model, um, bm, df_pairs, u_mean, b_mean, g_mean):
    model.eval()
    out = []
    with torch.no_grad():
        for _, row in df_pairs.iterrows():
            u = row["userID"]
            b = row["bookID"]
            if (u in um) and (b in bm):
                uu = torch.tensor([um[u]], dtype=torch.int64)
                bb = torch.tensor([bm[b]], dtype=torch.int64)
                v = model(uu, bb).item()
            else:
                v = g_mean
            if (u in u_mean) and (b in b_mean):
                v = 0.6 * v + 0.2 * u_mean[u] + 0.2 * b_mean[b]
            elif u in u_mean:
                v = 0.7 * v + 0.3 * u_mean[u]
            elif b in b_mean:
                v = 0.7 * v + 0.3 * b_mean[b]
            if v < 0:
                v = 0.0
            if v > 5:
                v = 5.0
            out.append(v)
    return np.array(out, dtype=np.float32)


def predict_rating_ensemble(models, ums, bms, df_pairs, u_mean, b_mean, g_mean):
    preds = []
    for m, um, bm in zip(models, ums, bms):
        p = predict_rating_single(m, um, bm, df_pairs, u_mean, b_mean, g_mean)
        preds.append(p)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0)


def learn_rating_stack(df_tr, df_val, param_list):
    u_mean = df_tr.groupby("userID")["rating"].mean().to_dict()
    b_mean = df_tr.groupby("bookID")["rating"].mean().to_dict()
    g_mean = float(df_tr["rating"].mean())

    batch = 4096
    models = []
    ums = []
    bms = []
    for i, (k, lr, wd, epochs) in enumerate(param_list):
        m, um, bm = train_rating_mf(df_tr, k, epochs, batch, lr, wd, seed=i)
        models.append(m)
        ums.append(um)
        bms.append(bm)

    df_val_pairs = df_val[["userID", "bookID"]].copy()
    y = df_val["rating"].values.astype(np.float32)

    feats = []
    for m, um, bm in zip(models, ums, bms):
        p = predict_rating_single(m, um, bm, df_val_pairs, u_mean, b_mean, g_mean)
        feats.append(p)
    feats = np.stack(feats, axis=1)

    u_feat = np.array([u_mean.get(u, g_mean) for u in df_val_pairs["userID"].values], dtype=np.float32)
    b_feat = np.array([b_mean.get(b, g_mean) for b in df_val_pairs["bookID"].values], dtype=np.float32)
    g_feat = np.full(len(df_val_pairs), g_mean, dtype=np.float32)

    X = np.column_stack([feats, u_feat, b_feat, g_feat])

    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    mse = float(np.mean((y_pred - y) ** 2))
    return reg.coef_.astype(np.float32), float(reg.intercept_)


def predict_rating_stacked(models, ums, bms, df_pairs, u_mean, b_mean, g_mean, w, b0):
    feats = []
    for m, um, bm in zip(models, ums, bms):
        p = predict_rating_single(m, um, bm, df_pairs, u_mean, b_mean, g_mean)
        feats.append(p)
    feats = np.stack(feats, axis=1)

    u_feat = np.array([u_mean.get(u, g_mean) for u in df_pairs["userID"].values], dtype=np.float32)
    b_feat = np.array([b_mean.get(b, g_mean) for b in df_pairs["bookID"].values], dtype=np.float32)
    g_feat = np.full(len(df_pairs), g_mean, dtype=np.float32)

    X = np.column_stack([feats, u_feat, b_feat, g_feat]).astype(np.float32)
    y = X @ w + b0
    y = np.clip(y, 0.0, 5.0)
    return y.astype(np.float32)


def make_read_pairs(df, n_neg=4, seed=0):
    pos = df[["userID", "bookID"]].copy()
    pos["label"] = 1.0

    user_items = df.groupby("userID")["bookID"].apply(set).to_dict()
    all_items = df["bookID"].unique()
    all_items = np.array(all_items)

    rng = np.random.default_rng(seed)
    neg_u = []
    neg_i = []
    neg_y = []

    pos_u = pos["userID"].to_numpy()
    pos_i = pos["bookID"].to_numpy()

    for u, b_pos in zip(pos_u, pos_i):
        items_u = user_items[u]
        for _ in range(n_neg):
            while True:
                b_neg = rng.choice(all_items)
                if b_neg not in items_u:
                    neg_u.append(u)
                    neg_i.append(b_neg)
                    neg_y.append(0.0)
                    break

    neg = pd.DataFrame(
        {"userID": neg_u, "bookID": neg_i, "label": np.array(neg_y, dtype=np.float32)}
    )

    df_pairs = pd.concat([pos, neg], ignore_index=True)
    return df_pairs


def build_read_stats(df):
    user_cnt = df.groupby("userID")["bookID"].count().to_dict()
    item_cnt = df.groupby("bookID")["userID"].count().to_dict()
    user_mean = df.groupby("userID")["rating"].mean().to_dict()
    item_mean = df.groupby("bookID")["rating"].mean().to_dict()
    global_mean = float(df["rating"].mean())
    return user_cnt, item_cnt, user_mean, item_mean, global_mean


def make_read_pairs_lr(df, n_neg=1, seed=0):
    return make_read_pairs(df, n_neg=n_neg, seed=seed)


def build_read_features(df_pairs, user_cnt, item_cnt, user_mean, item_mean, global_mean):
    u = df_pairs["userID"].values
    b = df_pairs["bookID"].values

    f_book_pop = np.log1p([item_cnt.get(bi, 0) for bi in b])
    f_user_act = np.log1p([user_cnt.get(ui, 0) for ui in u])
    f_user_mean = np.array([user_mean.get(ui, global_mean) for ui in u], dtype=np.float32)
    f_item_mean = np.array([item_mean.get(bi, global_mean) for bi in b], dtype=np.float32)

    f_user_bias = f_user_mean - global_mean
    f_item_bias = f_item_mean - global_mean

    X = np.column_stack(
        [
            f_book_pop,
            f_user_act,
            f_user_mean,
            f_item_mean,
            f_user_bias,
            f_item_bias,
        ]
    ).astype(np.float32)
    return X


def train_read_lr(df, n_neg=1, seed=0):
    np.random.seed(seed)

    user_cnt, item_cnt, user_mean, item_mean, global_mean = build_read_stats(df)

    pairs = make_read_pairs_lr(df, n_neg=n_neg, seed=seed)
    X = build_read_features(pairs, user_cnt, item_cnt, user_mean, item_mean, global_mean)
    y = pairs["label"].values.astype(int)

    idx_all = np.arange(len(y))
    tr_idx, val_idx = train_test_split(
        idx_all, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    p_val = clf.predict_proba(X_val)[:, 1]

    preds_05 = (p_val >= 0.5).astype(int)
    acc_05 = (preds_05 == y_val).mean()

    n_val = len(p_val)
    k = n_val // 2
    idx_top = np.argpartition(-p_val, k - 1)[:k]
    top_preds = np.zeros(n_val, dtype=int)
    top_preds[idx_top] = 1
    acc_top = (top_preds == y_val).mean()

    return clf, user_cnt, item_cnt, user_mean, item_mean, global_mean


def predict_read_lr(clf, df_pairs, user_cnt, item_cnt, user_mean, item_mean, global_mean):
    X = build_read_features(df_pairs, user_cnt, item_cnt, user_mean, item_mean, global_mean)
    p = clf.predict_proba(X)[:, 1]

    n = len(p)
    k = n // 2
    idx_top = np.argpartition(-p, k - 1)[:k]
    y_pred = np.zeros(n, dtype=int)
    y_pred[idx_top] = 1
    return y_pred


def load_cat_train(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            d = ast.literal_eval(t)
            rows.append(
                {
                    "user_id": d["user_id"],
                    "review_id": d["review_id"],
                    "text": d["review_text"],
                    "label": int(d["genreID"]),
                }
            )
    return pd.DataFrame(rows)


def load_cat_test(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            d = ast.literal_eval(t)
            rows.append(
                {
                    "user_id": d["user_id"],
                    "review_id": d["review_id"],
                    "text": d["review_text"],
                }
            )
    return pd.DataFrame(rows)


def train_bert(
    df_tr,
    df_val,
    df_te,
    model_name="roberta-base",
    max_len=256,
    batch=8,
    epochs=4,
    lr=2e-5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_name)

    y_tr = df_tr["label"].values
    y_val = df_val["label"].values
    x_tr = list(df_tr["text"].values)
    x_val = list(df_val["text"].values)
    x_te = list(df_te["text"].values)

    enc_tr = tok(x_tr, padding="max_length", truncation=True, max_length=max_len)
    enc_val = tok(x_val, padding="max_length", truncation=True, max_length=max_len)
    enc_te = tok(x_te, padding="max_length", truncation=True, max_length=max_len)

    ids_tr = torch.tensor(enc_tr["input_ids"], dtype=torch.long)
    mask_tr = torch.tensor(enc_tr["attention_mask"], dtype=torch.long)
    ids_val = torch.tensor(enc_val["input_ids"], dtype=torch.long)
    mask_val = torch.tensor(enc_val["attention_mask"], dtype=torch.long)
    ids_te = torch.tensor(enc_te["input_ids"], dtype=torch.long)
    mask_te = torch.tensor(enc_te["attention_mask"], dtype=torch.long)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    ds_tr = TensorDataset(ids_tr, mask_tr, y_tr_t)
    ds_val = TensorDataset(ids_val, mask_val)
    ds_te = TensorDataset(ids_te, mask_te)

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False)

    n_labels = len(np.unique(y_tr))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_labels,
    )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    soft = nn.Softmax(dim=1)

    epoch_val_probs = []
    epoch_te_probs = []
    epoch_val_acc = []

    best_state = None
    best_acc = -1.0

    for ep in range(epochs):
        model.train()
        for ids_b, mask_b, y_b in tqdm(dl_tr, desc=f"{model_name}_epoch_{ep+1}", leave=False):
            ids_b = ids_b.to(device)
            mask_b = mask_b.to(device)
            y_b = y_b.to(device)

            out = model(input_ids=ids_b, attention_mask=mask_b, labels=y_b)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for ids_b, mask_b in dl_val:
                ids_b = ids_b.to(device)
                mask_b = mask_b.to(device)
                out = model(input_ids=ids_b, attention_mask=mask_b)
                p = soft(out.logits).cpu().numpy()
                val_probs.append(p)
        val_probs = np.concatenate(val_probs, axis=0)
        y_pred = val_probs.argmax(axis=1)
        acc = accuracy_score(y_val, y_pred)

        epoch_val_probs.append(val_probs)
        epoch_val_acc.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        te_probs = []
        with torch.no_grad():
            for ids_b, mask_b in dl_te:
                ids_b = ids_b.to(device)
                mask_b = mask_b.to(device)
                out = model(input_ids=ids_b, attention_mask=mask_b)
                p = soft(out.logits).cpu().numpy()
                te_probs.append(p)
        te_probs = np.concatenate(te_probs, axis=0)
        epoch_te_probs.append(te_probs)

        print(f"[{model_name}] epoch {ep+1}/{epochs} val_acc = {acc:.4f}")

    return epoch_val_probs, epoch_te_probs, epoch_val_acc



def train_tfidf(df_tr, df_val, df_te):
    vec = TfidfVectorizer(
        max_features=200000,
        ngram_range=(1, 2),
        min_df=2,
    )
    x_tr = vec.fit_transform(df_tr["text"])
    y_tr = df_tr["label"].values
    clf = LogisticRegression(
        max_iter=300,
        C=4.0,
        n_jobs=-1,
    )
    clf.fit(x_tr, y_tr)
    x_val = vec.transform(df_val["text"])
    x_te = vec.transform(df_te["text"])
    p_val = clf.predict_proba(x_val)
    p_te = clf.predict_proba(x_te)
    return p_val, p_te


def run_category(train_path, test_path, out_path):
    df_all = load_cat_train(train_path)
    df_te = load_cat_test(test_path)

    df_tr, df_val = train_test_split(
        df_all,
        test_size=0.2,
        random_state=0,
        stratify=df_all["label"],
    )

    p_lr_val, p_lr_te = train_tfidf(df_tr, df_val, df_te)
    y_val = df_val["label"].values

    epoch_val_probs, epoch_te_probs, epoch_val_acc = train_bert(
        df_tr,
        df_val,
        df_te,
        model_name="roberta-base",
        max_len=384,
        batch=16,
        epochs=4,
        lr=1e-5,
    )

    alphas = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    best_global_acc = -1.0
    best_global_te_preds = None

    for ep_idx, (p_bert_val, p_bert_te) in enumerate(zip(epoch_val_probs, epoch_te_probs), start=1):

        best_a = 0.0
        best_acc = -1.0
        for a in alphas:
            p_mix_val = a * p_bert_val + (1.0 - a) * p_lr_val
            y_pred_val = p_mix_val.argmax(axis=1)
            acc = accuracy_score(y_val, y_pred_val)
            if acc > best_acc:
                best_acc = acc
                best_a = a

        print(f"[epoch {ep_idx}] best alpha = {best_a}, ensemble val_acc = {best_acc:.4f}")


        p_mix_te = best_a * p_bert_te + (1.0 - best_a) * p_lr_te
        y_te_ep = p_mix_te.argmax(axis=1)

 
        out_ep = pd.DataFrame(
            {
                "userID": df_te["user_id"],
                "reviewID": df_te["review_id"],
                "prediction": y_te_ep.astype(int),
            }
        )
        out_ep.to_csv(f"predictions_Category_epoch{ep_idx}.csv", index=False)

        if best_acc > best_global_acc:
            best_global_acc = best_acc
            best_global_te_preds = y_te_ep

    if best_global_te_preds is None:
        best_global_te_preds = y_te_ep

    out = pd.DataFrame(
        {
            "userID": df_te["user_id"],
            "reviewID": df_te["review_id"],
            "prediction": best_global_te_preds.astype(int),
        }
    )
    out.to_csv(out_path, index=False)
    print(f"Final best ensemble val_acc = {best_global_acc:.4f}")
    print(f"Wrote per-epoch files predictions_Category_epoch*.csv and final {out_path}")



def train_read_lr_rf(df, n_neg=1, seed=0):
    np.random.seed(seed)

    user_cnt, item_cnt, user_mean, item_mean, global_mean = build_read_stats(df)

    pairs = make_read_pairs_lr(df, n_neg=n_neg, seed=seed)
    X = build_read_features(pairs, user_cnt, item_cnt, user_mean, item_mean, global_mean)
    y = pairs["label"].values.astype(int)

    idx_all = np.arange(len(y))
    tr_idx, val_idx = train_test_split(
        idx_all, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    clf_lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        n_jobs=-1,
    )
    clf_lr.fit(X_tr, y_tr)

    clf_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    clf_rf.fit(X_tr, y_tr)

    p_val_lr = clf_lr.predict_proba(X_val)[:, 1]
    p_val_rf = clf_rf.predict_proba(X_val)[:, 1]

    n_val = len(p_val_lr)
    k = n_val // 2

    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_w = 0.5
    best_acc = -1.0
    for w in ws:
        p_val = w * p_val_lr + (1.0 - w) * p_val_rf
        idx_top = np.argpartition(-p_val, k - 1)[:k]
        preds_top = np.zeros(n_val, dtype=int)
        preds_top[idx_top] = 1
        acc = (preds_top == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_w = w

    return clf_lr, clf_rf, user_cnt, item_cnt, user_mean, item_mean, global_mean, best_w


def predict_read_lr_rf_peruser(clf_lr, clf_rf, df_pairs,
                               user_cnt, item_cnt, user_mean, item_mean, global_mean,
                               w):
    X = build_read_features(df_pairs, user_cnt, item_cnt, user_mean, item_mean, global_mean)

    p_lr = clf_lr.predict_proba(X)[:, 1]
    p_rf = clf_rf.predict_proba(X)[:, 1]
    scores = (w * p_lr + (1.0 - w) * p_rf).astype(np.float32)

    users = df_pairs["userID"].values
    n = len(scores)
    y_pred = np.zeros(n, dtype=int)

    groups = defaultdict(list)
    for idx, u in enumerate(users):
        groups[u].append(idx)

    for u, idxs in groups.items():
        idxs = np.array(idxs, dtype=int)
        m = len(idxs)
        if m <= 0:
            continue
        k = m // 2
        if k == 0:
            continue
        user_scores = scores[idxs]
        top_local = np.argpartition(-user_scores, k - 1)[:k]
        y_pred[idxs[top_local]] = 1

    return y_pred


def main():
    inter_path = "train_Interactions.csv.gz"
    pairs_rating_path = "pairs_Rating.csv"
    pairs_read_path = "pairs_Read.csv"
    train_cat_path = "train_Category.json.gz"
    test_cat_path = "test_Category.json.gz"


    # df_all = pd.read_csv(inter_path, compression="gzip")
    # df_all = df_all[["userID", "bookID", "rating"]]
    # df_all["rating"] = df_all["rating"].astype(np.float32)

    # df_tr, df_val = train_test_split(
    #     df_all,
    #     test_size=0.2,
    #     random_state=0,
    #     shuffle=True,
    # )

    # param_list = [
    #     (96, 0.01, 7e-05, 45),
    #     (128, 0.01, 5e-05, 45),
    #     (64, 0.01, 5e-05, 45),
    # ]

    # w_stack, b_stack = learn_rating_stack(df_tr, df_val, param_list)

    # u_mean = df_all.groupby("userID")["rating"].mean().to_dict()
    # b_mean = df_all.groupby("bookID")["rating"].mean().to_dict()
    # g_mean = float(df_all["rating"].mean())

    # models = []
    # ums = []
    # bms = []
    # batch = 4096
    # for i, (k, lr, wd, epochs) in enumerate(param_list):
    #     m, um, bm = train_rating_mf(df_all, k, epochs, batch, lr, wd, seed=i)
    #     models.append(m)
    #     ums.append(um)
    #     bms.append(bm)

    # df_pr = pd.read_csv(pairs_rating_path)[["userID", "bookID"]]
    # pred_r = predict_rating_stacked(models, ums, bms, df_pr, u_mean, b_mean, g_mean, w_stack, b_stack)
    # out_r = df_pr.copy()
    # out_r["prediction"] = pred_r
    # out_r.to_csv("predictions_Rating.csv", index=False)

    # clf_lr, clf_rf, user_cnt, item_cnt, user_mean, item_mean, g_mean, w_best = train_read_lr_rf(
    #     df_all,
    #     n_neg=1,
    #     seed=0,
    # )

    # df_read_test = pd.read_csv(pairs_read_path)[["userID", "bookID"]]
    # pred_read = predict_read_lr_rf_peruser(
    #     clf_lr,
    #     clf_rf,
    #     df_read_test.copy(),
    #     user_cnt,
    #     item_cnt,
    #     user_mean,
    #     item_mean,
    #     g_mean,
    #     w_best,
    # )

    # out_read = df_read_test.copy()
    # out_read["prediction"] = pred_read
    # out_read.to_csv("predictions_Read.csv", index=False)

    run_category(train_cat_path, test_cat_path, "predictions_Category.csv")


if __name__ == "__main__":
    main()