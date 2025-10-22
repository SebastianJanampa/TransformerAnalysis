import os
import torch
import numpy as np

def thresholding(eigenvals,  thresholding):
    max_sigma = eigenvals[0]
    n = len(eigenvals) - 1
    cond_num = max_sigma / eigenvals[n]
    while n > 4 and cond_num > thresholding:
        n-=1
        cond_num= max_sigma / eigenvals[n]
    eigenvals = eigenvals[:n+1]
    return eigenvals, n+1

def extract_spaces(A):
    rank = np.linalg.matrix_rank(A)
    U, S, Vh = np.linalg.svd(A)
    S = S
    row_space = Vh[:rank, :].T
    null_space = Vh[rank: :].T
    col_space = U[:, :rank]
    left_null = U[:, :rank]

    return [row_space, null_space, col_space, left_null], [U, S, Vh], rank

def eigenval_modification(model, print_info=True, threshold=10):

    if os.path.exists(f'./DINO_{threshold}.pth'):
        model.load_state_dict(torch.load(f'./DINO_{threshold}.pth', weights_only=True))
    else:
        device =  next(model.parameters()).device
        num_heads = model.num_heads
        embed_dim = model.embed_dim

        qkv_layer_weights, proj_layer_weights = [], []
        for block in model.blocks:
            weights = block.attn.qkv.weight.detach() # (out_feats, in_feats) 
            weights = weights.view(num_heads, 3, embed_dim // num_heads, embed_dim)
            qkv_layer_weights.append(weights)

            weights = block.attn.proj.weight.detach()
            proj_layer_weights.append(weights)

        for block_id in range(len(qkv_layer_weights)):
            # For qkv layer
            layer_weights, new_weights_block = qkv_layer_weights[block_id], []

            for i in range(3):
                block = layer_weights[:, i, :, :].cpu().numpy() 

                if print_info:
                    print("#"*36 + f"Transformer Block {block_id:02d} -- {i+1}/3" + "#"*37)

                new_weights = []
                for head_id, head  in enumerate(block, start=1):
                    space_head, svd_head, rank = extract_spaces(head)
                    U, s, Vh = svd_head
                    s, n = thresholding(s, threshold)
                    S = np.zeros((U.shape[1], Vh.shape[0]), dtype=U.dtype)
                    S[:n, :n] = np.diag(s)
                    new_weights_head = U @ S @ Vh
                    new_weights.append(new_weights_head)

                    max_sigma = s.max()
                    min_sigma = s.min()
                    if print_info:
                        print("="*36 + f"Head {head_id:02d}" + "="*37)
                        print(f"rank:{rank}     new num of σs: {n}  max σ: {max_sigma:.3f}   min σ: {min_sigma:.3f}   (max σ / min σ): {max_sigma/min_sigma:.3f}")
                new_weights = np.stack(new_weights, axis=0) 
                new_weights_block.append(new_weights)

            new_weights_block = np.stack(new_weights_block, axis=1) 
            new_weights_block = new_weights_block.reshape(-1, embed_dim) # orginal shape for torch.nn.Linear layer
            
            model.blocks[block_id].attn.qkv.weight.data = torch.tensor(new_weights_block, device=device) 

            # For projection layer
            layer_weights = proj_layer_weights[block_id].cpu().numpy() 
            space_head, svd_head, rank = extract_spaces(layer_weights)
            U, s, Vh = svd_head
            s, n = thresholding(s, threshold)
            S = np.zeros((U.shape[1], Vh.shape[0]), dtype=U.dtype)
            S[:n, :n] = np.diag(s)
            new_weights = U @ S @ Vh

            max_sigma = s.max()
            min_sigma = s.min()

            if print_info:
                print("#"*36 + f"Transformer Block {block_id:02d} - Proj" + "#"*37)
                print(f"rank:{rank}     new num of σs: {n}  max σ: {max_sigma:.3f}   min σ: {min_sigma:.3f}   (max σ / min σ): {max_sigma/min_sigma:.3f}")

            model.blocks[block_id].attn.proj.weight.data = torch.tensor(new_weights, device=device) 

        print(f"Saving DINO_{threshold}.pth")
        torch.save(model.state_dict(), f'DINO_{threshold}.pth')
        
    return model