from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()

def add_privacy(model, optimizer, trainset, noise_multiplier):
  model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=trainset,
    noise_multiplier=noise_multiplier,
    max_grad_norm=1.0,
  )
  
  return model, optimizer, data_loader