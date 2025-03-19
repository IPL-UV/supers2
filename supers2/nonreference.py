import torch


def srmodel(
    sr_model: torch.nn.Module,
    hard_constraint: torch.nn.Module,
    device: str = "cpu",
):
    # Load the SR model    
    sr_model.to(device)

    # Load HardConstraint
    hard_constraint = hard_constraint.eval()
    for param in hard_constraint.parameters():
        param.requires_grad = False
    hard_constraint = hard_constraint.to(device)

    class SRModel(torch.nn.Module):
        def __init__(self, sr_model, hard_constraint):
            super().__init__()
            self.sr_model = sr_model
            self.hard_constraint = hard_constraint

        def forward(self, x):
            sr = self.sr_model(x)
            return self.hard_constraint(x, sr)
    
    return SRModel(sr_model, hard_constraint)