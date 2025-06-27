from __future__ import absolute_import, division, print_function
import argparse
from math import log
from tqdm import tqdm
import torch
import multiprocessing as mp
from functools import reduce
from kg_env import BatchKGEnvironment, BatchCFKGEnvironment
from train_agent import ActorCritic
from utils import *


def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_ids = list(test_user_products.keys())
    
    for user_id in test_user_ids:
        if user_id not in topk_matches or len(topk_matches[user_id]) < 10:
            invalid_users.append(user_id)
            continue
        
        pred_list, relevant_set = topk_matches[user_id][::-1], test_user_products[user_id]
        if not pred_list:
            continue

        dcg, hit_count = 0.0, 0.0
        for i, pred in enumerate(pred_list):
            if pred in relevant_set:
                dcg += 1.0 / (log(i + 2) / log(2))
                hit_count += 1
        
        idcg = sum(1.0 / (log(i + 2) / log(2)) for i in range(min(len(relevant_set), len(pred_list))))
        ndcg = dcg / idcg
        recall = hit_count / len(relevant_set)
        precision = hit_count / len(pred_list)
        hit = 1.0 if hit_count > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print(f"NDCG={avg_ndcg:.3f} | Recall={avg_recall:.3f} | HR={avg_hit:.3f} | Precision={avg_precision:.3f} | Invalid users={len(invalid_users)}")


def batch_beam_search(env, model, user_ids, device, dataset, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        return np.vstack([np.pad(np.ones(len(acts)), (0, model.act_dim - len(acts)), 'constant') for acts in batch_acts])

    state_pool = env.reset(user_ids)
    path_pool = env._batch_path
    probs_pool = [[] for _ in user_ids]
    model.eval()
    
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)
        actmask_pool = _batch_acts_to_masks(acts_pool)
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))
        probs += actmask_tensor.float()
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)
        topk_idxs, topk_probs = topk_idxs.cpu().numpy(), topk_probs.cpu().detach().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path, probs = path_pool[row], probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):
                    continue
                relation, next_node_id = acts_pool[row][idx]
                next_node_type = path[-1][1] if relation == SELF_LOOP else (FOOD_KG_RELATION if dataset == FOOD else KG_RELATION)[path[-1][1]][relation]
                new_path_pool.append(path + [(relation, next_node_type, next_node_id)])
                new_probs_pool.append(probs + [p])
        
        path_pool, probs_pool = new_path_pool, new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def _predict_paths_worker(args_bundle):
    """Worker function for predict_paths."""
    worker_id, user_ids_chunk, policy_file, args = args_bundle
    
    # Always use device 0 for all workers
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    pretrain_sd = torch.load(policy_file, map_location=device)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(device)
    model.load_state_dict({**model.state_dict(), **pretrain_sd})
    model.eval()

    batch_size = 16
    paths_chunk, probs_chunk = [], []
    for i in range(0, len(user_ids_chunk), batch_size):
        batch_user_ids = user_ids_chunk[i:i + batch_size]
        paths, probs = batch_beam_search(
            env, model, batch_user_ids, device, dataset=args.dataset, topk=args.topk)
        paths_chunk.extend(paths)
        probs_chunk.extend(probs)
    return paths_chunk, probs_chunk


def predict_paths(policy_file, args, path_file):
    print("Predicting paths...")
    test_labels = load_labels(args.dataset, "test")
    test_user_ids = list(test_labels.keys())

    num_workers = max(1, min(args.num_workers, mp.cpu_count()))
    chunk_size = (len(test_user_ids) + num_workers - 1) // num_workers
    user_id_chunks = [test_user_ids[i:i + chunk_size] for i in range(0, len(test_user_ids), chunk_size)]
    args_bundles = [(i, chunk, policy_file, args) for i, chunk in enumerate(user_id_chunks)]

    all_paths, all_probs = [], []
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(_predict_paths_worker, args_bundles), total=len(args_bundles), desc="Predicting Paths"))

    for paths_chunk, probs_chunk in results:
        all_paths.extend(paths_chunk)
        all_probs.extend(probs_chunk)
        
    pickle.dump({"paths": all_paths, "probs": all_probs}, open(path_file, "wb"))


def _evaluate_best_paths_worker(args_bundle):
    """Worker function to find the best path for each user-product pair."""
    user_id_chunk, pred_paths, train_labels = args_bundle
    best_pred_paths_chunk = {}
    for user_id in user_id_chunk:
        if user_id not in pred_paths:
            continue
        best_pred_paths_chunk[user_id] = []
        train_product_ids = set(train_labels.get(user_id, []))
        for product_id in pred_paths[user_id]:
            if product_id in train_product_ids:
                continue
            sorted_paths = sorted(pred_paths[user_id][product_id], key=lambda x: x[1], reverse=True)
            if sorted_paths:
                best_pred_paths_chunk[user_id].append(sorted_paths[0])
    return best_pred_paths_chunk


def _evaluate_labels_worker(args_bundle):
    """Worker function to generate final sorted recommendation lists."""
    user_id_chunk, best_pred_paths, scores, train_labels, args = args_bundle
    pred_labels_chunk = {}
    for user_id in user_id_chunk:
        if user_id not in best_pred_paths or not best_pred_paths[user_id]:
            pred_labels_chunk[user_id] = []
            continue
        
        sorted_paths = sorted(best_pred_paths[user_id], key=lambda x: (x[0], x[1]), reverse=True)
        top10_product_ids = [p[-1][2] for _, _, p in sorted_paths[:10]]
        
        if args.add_products and len(top10_product_ids) < 10:
            train_product_ids = set(train_labels.get(user_id, []))
            candidate_product_ids = np.argsort(scores[user_id])
            for candidate_product_id in candidate_product_ids[::-1]:
                if len(top10_product_ids) >= 10:
                    break
                if candidate_product_id in train_product_ids or candidate_product_id in top10_product_ids:
                    continue
                top10_product_ids.append(candidate_product_id)
        
        pred_labels_chunk[user_id] = top10_product_ids[::-1]
    return pred_labels_chunk


def evaluate_paths(path_file, train_labels, test_labels, args):
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    purchase_embeds = embeds[VIEW][0] if args.dataset == FOOD else embeds[PURCHASE][0]
    product_embeds = embeds[RECIPE] if args.dataset == FOOD else embeds[PRODUCT]
    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

    results = pickle.load(open(path_file, "rb"))
    pred_paths = {uid: {} for uid in test_labels}
    
    for path, probs in zip(results["paths"], results["probs"]):
        if path[-1][1] not in {PRODUCT, RECIPE}:
            continue
        user_id, product_id = path[0][2], path[-1][2]
        if user_id not in pred_paths:
            continue
        if product_id not in pred_paths[user_id]:
            pred_paths[user_id][product_id] = []
        path_score = scores[user_id][product_id]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[user_id][product_id].append((path_score, path_prob, path))
    
    # Parallelize finding best paths
    num_workers = max(1, min(args.num_workers, mp.cpu_count()))
    ctx = mp.get_context('spawn')
    user_ids = list(pred_paths.keys())
    chunk_size = (len(user_ids) + num_workers - 1) // num_workers
    user_id_chunks = [user_ids[i:i + chunk_size] for i in range(0, len(user_ids), chunk_size)]
    args_bundles_1 = [(chunk, pred_paths, train_labels) for chunk in user_id_chunks]
    
    best_pred_paths = {}
    with ctx.Pool(processes=num_workers) as pool:
        results_1 = list(tqdm(pool.imap_unordered(_evaluate_best_paths_worker, args_bundles_1), total=len(args_bundles_1), desc="Finding best paths"))
    for chunk_result in results_1:
        best_pred_paths.update(chunk_result)

    # Parallelize generating final labels
    user_ids_for_labels = list(best_pred_paths.keys())
    chunk_size_2 = (len(user_ids_for_labels) + num_workers - 1) // num_workers
    user_id_chunks_2 = [user_ids_for_labels[i:i + chunk_size_2] for i in range(0, len(user_ids_for_labels), chunk_size_2)]
    args_bundles_2 = [(chunk, best_pred_paths, scores, train_labels, args) for chunk in user_id_chunks_2]

    pred_labels = {}
    with ctx.Pool(processes=num_workers) as pool:
        results_2 = list(tqdm(pool.imap_unordered(_evaluate_labels_worker, args_bundles_2), total=len(args_bundles_2), desc="Generating labels"))
    for chunk_result in results_2:
        pred_labels.update(chunk_result)

    evaluate(pred_labels, test_labels)


def test(args):
    policy_file = f"{args.log_dir}/policy_model_epoch_{args.epochs}.ckpt"
    path_file = f"{args.log_dir}/policy_paths_epoch{args.epochs}.pkl"

    train_labels = load_labels(args.dataset, "train")
    test_labels = load_labels(args.dataset, "test")

    if args.run_path:
        predict_paths(policy_file, args, path_file)
    if args.run_eval:
        evaluate_paths(path_file, train_labels, test_labels, args)


if __name__ == "__main__":
    boolean = lambda x: str(x).lower() == "true"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=CELL, help="One of {cloth, beauty, cell, cd}")
    parser.add_argument("--name", type=str, default="train_agent", help="directory name.")
    parser.add_argument("--seed", type=int, default=123, help="random seed.")
    parser.add_argument("--gpu", type=str, default="0", help="gpu device.")
    parser.add_argument("--epochs", type=int, default=50, help="num of epochs.")
    parser.add_argument("--max_acts", type=int, default=100, help="Max number of actions.")
    parser.add_argument("--max_path_len", type=int, default=3, help="Max path length.")
    parser.add_argument("--gamma", type=float, default=0.99, help="reward discount factor.")
    parser.add_argument("--state_history", type=int, default=1, help="state history length")
    parser.add_argument("--hidden", type=int, nargs="*", default=[512, 256], help="hidden layer sizes.")
    parser.add_argument("--add_products", type=boolean, default=True, help="Add predicted products up to 10")
    parser.add_argument("--topk", type=int, nargs="*", default=[25, 5, 1], help="number of samples")
    parser.add_argument("--run_path", action="store_true", help="Run path prediction?")
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation?")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of workers for multiprocessing.")
    
    args = parser.parse_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args.log_dir = f"{TMP_DIR[args.dataset]}/{args.name}"
    
    test(args)