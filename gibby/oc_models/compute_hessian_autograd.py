import torch


def compute_hessian_autograd(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:

    hessian = []
    for grad_elem in forces.contiguous().view(-1):
        hess_row = (
            -1
            * torch.autograd.grad(
                outputs=[grad_elem],
                inputs=[positions],
                grad_outputs=torch.ones_like(grad_elem),
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
        )
        hess_row = hess_row.detach()
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)

    return hessian
