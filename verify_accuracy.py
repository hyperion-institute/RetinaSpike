feature_net = FeatureExtractor().to(device)
snn_model = SNNModel().to(device)

checkpoint = torch.load("RetinaSpike.pth", map_location=device)

feature_net.load_state_dict(checkpoint["feature_net"])
snn_model.load_state_dict(checkpoint["snn_model"])

feature_net.eval()
snn_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        f = feature_net(data)
        contrast = compute_contrast(f)

        spike = encode_to_spike(
            contrast,
            T=checkpoint["T"],
            threshold=checkpoint["threshold"]
        )

        output, _ = snn_model(spike)

        pred = output.argmax(dim=1)

        correct += (pred == target).sum().item()
        total += target.size(0)

acc = 100 * correct / total
print("Loaded model accuracy:", acc)
