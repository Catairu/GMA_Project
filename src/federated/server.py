from collections import OrderedDict
import torch


class FedAvgServer:
    def __init__(self, model_fn, device: str = "cpu") -> None:
        self.model_fn = model_fn
        self.device = torch.device(device)
        self.global_model = model_fn().to(self.device)

    def get_global_state(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in self.global_model.state_dict().items()
        )

    def aggregate(
        self, client_updates: list[tuple[OrderedDict[str, torch.Tensor], int]]
    ) -> None:
        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples == 0:
            return

        first_state = client_updates[0][0]
        agg_state: OrderedDict[str, torch.Tensor] = OrderedDict(
            (k, torch.zeros_like(v, dtype=torch.float32)) for k, v in first_state.items()
        )

        for state_dict, num_samples in client_updates:
            weight = num_samples / total_samples
            for key in agg_state:
                agg_state[key] += state_dict[key].float() * weight

        self.global_model.load_state_dict(agg_state, strict=True)
