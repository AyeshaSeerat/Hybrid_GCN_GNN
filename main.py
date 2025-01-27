import torch
import argparse
from torch_geometric.loader import DataLoader
from dataset import load_data_from_npz, GraphDataset
from train import train, evaluate
from model import GNN, GCN, CombinedGCNGNN

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run Combined GCN + GNN Model")
    parser.add_argument('--gcn_hidden', type=int, default=64, help="Hidden dimension for GCN")
    parser.add_argument('--gnn_hidden', type=int, default=64, help="Hidden dimension for GNN")
    parser.add_argument('--out_channels', type=int, default=4, help="Output dimension (number of classes)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument('--dataset_name', type=str, default="Twitter15", help="Dataset name for logging and results")
    parser.add_argument('--save_model', action='store_true', help="Whether to save the best model")
    parser.add_argument('--model_path', type=str, default="combined_model.pth", help="Path to save the best model")
    args = parser.parse_args()

    # Load dataset
    data_dir = f'gen/{args.dataset_name}graph15'
    data_list = load_data_from_npz(data_dir)

    # Split dataset into training, validation, and testing
    train_size = int(0.7 * len(data_list))
    val_size = int(0.15 * len(data_list))
    test_size = len(data_list) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(data_list, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(GraphDataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(GraphDataset(val_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=args.batch_size, shuffle=False)

    # Initialize the GCN and simple GNN models
    gcn = GCN(in_channels=5000, hidden_channels=args.gcn_hidden, out_channels=args.gcn_hidden, dropout_rate=0.6)
    gnn = GNN(in_channels=args.gcn_hidden, hidden_channels=args.gnn_hidden, out_channels=args.out_channels, dropout_rate=0.5)

    # Combine GCN and GNN
    combined_model = CombinedGCNGNN(
        gcn=gcn,
        gnn=gnn,
        gcn_output_dim=args.gcn_hidden,
        hidden_channels=args.gnn_hidden,
        out_channels=args.out_channels
    )

    # Print selected configuration
    print("=========== Training Configuration ===========")
    print(f"GCN Hidden Dimension: {args.gcn_hidden}")
    print(f"GNN Hidden Dimension: {args.gnn_hidden}")
    print(f"Output Channels: {args.out_channels}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Dataset Name: {args.dataset_name}")
    if args.save_model:
        print(f"Model Save Enabled: True")
        print(f"Model Save Path: {args.model_path}")
    else:
        print(f"Model Save Enabled: False")
    print("==============================================")

    # Debugging edge index and node features
    for batch_idx, data in enumerate(train_loader):
        max_index = data.edge_index.max()
        node_count = data.x.shape[0]
        if max_index >= node_count:
            print(f"Error in batch {batch_idx}: Max index in edge_index: {max_index}, Node count: {node_count}")
            raise ValueError("Edge index contains invalid references.")

    # Train the combined model
    train(
        model=combined_model, #combined_model,gcn,gnn
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=5,
        dataset_name=args.dataset_name,
        save_best_model=args.save_model,
        best_model_path=args.model_path
    )

    # Evaluate the combined model on the test set
    evaluate(combined_model, test_loader, dataset_name=args.dataset_name)
