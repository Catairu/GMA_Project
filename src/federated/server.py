from collections import OrderedDict
import torch


class FedAvgServer:
    def __init__(self, model_fn, device: str = "cpu") -> None:
        self.model_fn = model_fn
        self.device = torch.device(device)
        self.global_model = model_fn().to(self.device)
        self.current_client_updates: dict[int, OrderedDict[str, torch.Tensor]] | None = None 

    def get_global_state(self) -> OrderedDict[str, torch.Tensor]:
        """Returns a detached CPU copy of the global model state dict."""
        return OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in self.global_model.state_dict().items()
        )
    def get_local_states(self) -> dict[int, OrderedDict[str, torch.Tensor]]:
        """Returns the most recent client updates as {client_id: state_dict}."""
        return self.current_client_updates
    
    def aggregate(
        self, client_updates: list[tuple[int, OrderedDict[str, torch.Tensor], int]]
    ) -> None:
        total_samples = sum(num_samples for _, _, num_samples in client_updates)
        if total_samples == 0:
            return

        first_state = client_updates[0][1]
        agg_state: OrderedDict[str, torch.Tensor] = OrderedDict(
            (k, torch.zeros_like(v, dtype=torch.float32)) for k, v in first_state.items()
        )
        
        new_state_dicts: dict[int, OrderedDict[str, torch.Tensor]] = {} # client_id -> state_dict
        for client_id, state_dict, num_samples in client_updates:
            new_state_dicts[client_id] = state_dict
            weight = num_samples / total_samples
            for key in agg_state:
                agg_state[key] += state_dict[key].float() * weight

        self.current_client_updates = new_state_dicts
        self.global_model.load_state_dict(agg_state, strict=True)
