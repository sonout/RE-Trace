import torch
from model.traj2vec import Traj2VecModel
import os
from eval_utils import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
from tqdm import tqdm

from eval_utils import Dataset


def main(args):

    outputname = args.t_filename + f"_s_retrace.pkl"
    save_output_path = os.path.join(args.data_path, outputname)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')    
    
    # Init Validation Setup
    dataset = Dataset(os.path.join(args.data_path, args.t_filename + ".t"), args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = Traj2VecModel(hidden_dim=args.hidden_dim,emb_dim=args.emb_dim, vocab_size=args.vocab_size, device=device)
    model.load_model(args.save_model_path)
    
    features_list = []
    with torch.no_grad():
        model.model.eval()
        for batch in tqdm(dataloader):
            trajs, trajs_len = [b.to(device) for b in batch]
            # batch_size, hidden_size * n_direction
            features = model.model.embed_trajectory(trajs, trajs_len)


            features_list.append(features.cpu())
        features_list = torch.cat(features_list, dim=0).numpy()

    pickle.dump(features_list, open(save_output_path, "wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='traj2vec')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--data_path', default="/data/shared/schestakov/re-identification/data/porto/db/dl_models", type=str)
    #parser.add_argument('--data_path', default="/home/schestakov/data/re-identification/sf/preprocessed", type=str)    
    parser.add_argument('--save_model_path', default="/data/shared/schestakov/re-identification/model_retrace/states/porto/ablation/no_pool", type=str)
    #parser.add_argument('--save_model_path', default="/home/schestakov/projects/re-identification/model_contrastive2/saved_model/sf", type=str)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--vocab_size', default=18866, type=int) # 18866 porto, 34109 sf
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--cell_rep_path', default="/home/schestakov/projects/re-identification/model_retrace/cell_rep/porto_128_gridsize_100.txt", type=str)
    #parser.add_argument('--cell_rep_path', default="/home/schestakov/projects/re-identification/model_cst-sim/pretrained/sf_128_gridsize_100.txt", type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=int)
    # Evaluation Arguments
    parser.add_argument('--t_filename', default="total", type=str)


    args = parser.parse_args()

    main(args)

