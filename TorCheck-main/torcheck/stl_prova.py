 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================

"""A fully-differentiable implementation of Signal Temporal Logic semantic trees."""

from typing import Union

# For custom type-hints
# For tensor functions
import torch
import torch.nn.functional as F
from torch import Tensor

# Custom types
realnum = Union[float, int] # Defines a custom type alias for any real number (used for thresholds)


# TODO: automatic check of timespan when evaluating robustness? (should be done only at root node)

def eventually(x: Tensor, time_span: int) -> Tensor:
    # TODO: as of this implementation, the time_span must be int (we are working with steps,
    #  not exactly points in the time axis)
    # TODO: maybe converter from resolution to steps, if one has different setting
    """
    STL operator 'eventually' in 1D.

    Parameters
    ----------
    x: torch.Tensor
        Signal
    time_span: any numeric type
        Timespan duration

    Returns
    -------
    torch.Tensor
    A tensor containing the result of the operation.
    """
    return F.max_pool1d(x, kernel_size=time_span, stride=1)
    # Defines an eventually operator using 1D max pooling:
    #    Applies over time.
    #    For each window of time_span, takes the maximum value.
    #    Used in quantitative semantics to implement F and G

class Node:
    """Abstract node class for STL semantics tree."""

    def __init__(self) -> None:
        # Must be overloaded.
        pass

    def __str__(self) -> str:
        # Must be overloaded.
        pass

    def boolean(self, x: Tensor, evaluate_at_all_times: bool = False) -> Tensor:
        """
        Evaluates the boolean semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with trhee dimensions.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the boolean semantics for the node.
        """
        z: Tensor = self._boolean(x)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

        # Calls internal _boolean method to evaluate Boolean semantics.
        # By default returns value at time t=0 unless evaluate_at_all_times=True.

    def quantitative(
            self,
            x: Tensor,
            normalize: bool = False,
            evaluate_at_all_times: bool = False,
    ) -> Tensor:
        """
        Evaluates the quantitative semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with three dimensions.
        normalize: bool
            Whether the measure of robustness if normalized (True) or
            not (False). Currently not in use.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the quantitative semantics for the node.
        """
        z: Tensor = self._quantitative(x, normalize) # normalize is a placeholder
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)
        
        # Same idea as boolean(...) but for robustness evaluation.
        
    def set_normalizing_flag(self, value: bool = True) -> None:
        """
        Setter for the 'normalization of robustness of the formula' flag.
        Currently not in use.
        """

    def time_depth(self) -> int:
        """Returns time depth of bounded temporal operators only."""
        # Must be overloaded.

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    def _boolean(self, x: Tensor) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        """Extrapolates the vector of truth values at time zero"""
        return torch.reshape(x[:, 0, 0], (-1,))


class Atom(Node):
    """Atomic formula node; for now of the form X<=t or X>=t"""

    def __init__(self, var_index: int, threshold: realnum, lte: bool = False) -> None:
        super().__init__()
        self.var_index: int = var_index
        self.threshold: realnum = threshold
        self.lte: bool = lte

    def __str__(self) -> str:
        s: str = (
                "x_"
                + str(self.var_index)
                + (" <= " if self.lte else " >= ")
                + str(round(self.threshold, 4))
        )
        return s

    def time_depth(self) -> int:
        return 0

    def _boolean(self, x: Tensor) -> Tensor: # Boolean: checks if x[var_index] <= threshold
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = torch.le(xj, self.threshold)
        else:
            z: Tensor = torch.ge(xj, self.threshold)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: # Quantitative: returns the signed distance from the threshold (e.g. how much above or below)
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = -xj + self.threshold
        else:
            z: Tensor = xj - self.threshold
        if normalize:
            z: Tensor = torch.tanh(z)
        return z


class Not(Node):
    """Negation node."""

    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child: Node = child

    def __str__(self) -> str:
        s: str = "not ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        return self.child.time_depth()

    def _boolean(self, x: Tensor) -> Tensor:
        z: Tensor = ~self.child._boolean(x)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z: Tensor = -self.child._quantitative(x, normalize) # negates the robustness score
        return z


class And(Node):
    """Conjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " and "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_and(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.min(z1, z2)
        return z


class Or(Node):
    """Disjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " or "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_or(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.max(z1, z2)
        return z


class Globally(Node): # ALWAYS operator: the formula is always true in the time span
    """Globally node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "always" + s0 + " ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])  # nested temporal parameters
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(1.0 - eventually((~z1).double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = -eventually(-z1, self.right_time_bound - self.left_time_bound)
        return z


class Eventually(Node): # EVENTUALLY operator: the formula is eventually true (true at any time) in the time span
    """Eventually node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "eventually" + s0 + " ( " + self.child.__str__() + " )"
        return s

    # TODO: coherence between computation of time depth and time span given when computing eventually 1d
    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(eventually(z1.double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
        return z


class Until(Node):
    # UNTIL operator: phi_1 U[a, b] phi_2: 
    #   phi_2 must be true sometime in [a, b]
    #   phi_1 must be true at all time before that
    
    # TODO: maybe define timed and untimed until, and use this class to wrap them
    # TODO: maybe faster implementation (of untimed until especially)
    """Until node."""

    def __init__(
            self,
            left_child: Node, # phi_1
            right_child: Node, # phi_2
            unbound: bool = False, # phi_1 U phi_2 over infinite time
            right_unbound: bool = False, # only upper bound is infinite
            left_time_bound: int = 0, # define the interval [a, b]
            right_time_bound: int = 1, # define the interval [a, b]
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound # Unbounded: full future
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( " + self.left_child.__str__() + " until" + s0 + " " + self.right_child.__str__() + " )"
        return s

    def time_depth(self) -> int: # Calculates how far in time this operator needs to look to compute its output
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return sum_children_depth + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        if self.unbound:
            # this is phi_1 U phi_2 over the full future: 
            # We compute, for each time step, whether phi_2 becomes true in the future, and phi_1 holds up until then 
            z1: Tensor = self.left_child._boolean(x) # Get Boolean values for both operands
            z2: Tensor = self.right_child._boolean(x)
            # Shape: [batch, 1, time]
            size: int = min(z1.size()[2], z2.size()[2]) 
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]
            
            # Builds a "history matrix" for each time step: was phi_1 true from now until t?
            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]
            # same for z2
            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            
            # Compute min(phi_1 segment, phi_2) across time window. Then take max over possible satisfaction points
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0] # This gives the truth value of the Until for each time point.
        elif self.right_unbound:
            # φ1 U[a,b] φ2 ≡ G[0,a](φ1) ∧ F[a,b](φ2) ∧ F[a,b](φ1 U φ2)
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
            # It rewrites bounded until in terms of:
            #   Globally: ensures φ1 holds at start
            #   Eventually(φ2): ensures φ2 happens
            #   Eventually(UnboundedUntil): handles general satisfaction
        
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._quantitative(x, normalize)
            z2: Tensor = self.right_child._quantitative(x, normalize)
            size: int = min(z1.size()[2], z2.size()[2])
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]

            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
            # z: Tensor = torch.cat([torch.max(torch.min(
            #    torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1),
            #    dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
        elif self.right_unbound:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        return z

class Since(Node):
    """Since node."""

    # SINCE operator: phi_1 U[a, b] phi_2: 
    #   phi_2 held within [a, b]
    #   phi_1 held from then until now

    # STL doesn’t natively support past, but we can simulate it by flipping time and reusing Until
    
    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (not self.unbound) and (not self.right_unbound) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = f"( {self.left_child} since{s0} {self.right_child} )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            return sum_children_depth + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [2]) # reverse along time axis

        # Reuse Until semantics on flipped signal
        until_node = Until( # construct a matching Until
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        # Compute on flipped signal
        z_flipped = until_node._boolean(x_flipped)

        # Flip back
        return torch.flip(z_flipped, [2]) # flip back

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [2])

        until_node = Until(
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        z_flipped = until_node._quantitative(x_flipped, normalize=normalize)

        return torch.flip(z_flipped, [2])

class DynamicReach(Node):
    def __init__(
        self,
        phi1: Node,
        phi2: Node,
        distance_fn: Callable[[Tensor], Tensor],
        d1: float,
        d2: Optional[float] = None,
        beta: float = 10.0,
        max_iter: int = 10,
    ):
        super().__init__()
        self.phi1 = phi1
        self.phi2 = phi2
        self.distance_fn = distance_fn
        self.d1 = d1
        self.d2 = float('inf') if d2 is None else d2
        self.beta = beta
        self.max_iter = max_iter

    def __str__(self):
        return f"DynamicReach({self.phi1}, {self.phi2}, d1={self.d1}, d2={self.d2})"

    def time_depth(self):
        return max(self.phi1.time_depth(), self.phi2.time_depth())

    def _soft_distance_matrix(self, edge_index, edge_dists, N):
        device = edge_dists.device
        dist = torch.full((N, N), float("inf"), device=device)
        dist.fill_diagonal_(0)
        src, dst = edge_index
        dist[src, dst] = edge_dists

        for _ in range(self.max_iter):
            current = dist.unsqueeze(0) + dist.unsqueeze(1)
            softmin = -torch.logsumexp(-self.beta * current, dim=2) / self.beta
            dist = torch.minimum(dist, softmin)
        return dist

    def _build_soft_adjacency(self, phi1_val, edge_index, edge_dists, N):
        src, dst = edge_index
        A = torch.zeros(N, N, device=phi1_val.device)
        A[src, dst] = torch.exp(-self.beta * edge_dists).clamp(min=1e-6)
        gate = torch.sigmoid(self.beta * phi1_val)
        return A.unsqueeze(0) * gate.unsqueeze(1) * gate.unsqueeze(2)

    def _quantitative(self, x, edge_index_seq, edge_weight_seq, normalize=False):
        phi1_val = self.phi1._quantitative(x, normalize).squeeze(1)
        phi2_val = self.phi2._quantitative(x, normalize).squeeze(1)
        B, N, T = phi1_val.shape

        results = []
        for t in range(T):
            edge_index = edge_index_seq[t]
            edge_weights = edge_weight_seq[t]
            edge_dists = self.distance_fn(edge_weights)
            soft_dist = self._soft_distance_matrix(edge_index, edge_dists, N)

            mask1 = torch.sigmoid(self.beta * (soft_dist - self.d1))
            mask2 = torch.sigmoid(self.beta * (soft_dist - self.d2))
            dist_mask = mask1 * (1 - mask2)

            phi1_t = phi1_val[:, :, t]
            phi2_t = phi2_val[:, :, t]
            soft_adj = self._build_soft_adjacency(phi1_t, edge_index, edge_dists, N)

            score = phi2_t.clone()
            for _ in range(self.max_iter):
                score = torch.maximum(score, torch.bmm(soft_adj, score.unsqueeze(-1)).squeeze(-1))

            denom = dist_mask.sum(1, keepdim=True).clamp(min=1e-6)
            masked_score = torch.bmm(dist_mask[None].expand(B, -1, -1), score.unsqueeze(-1)).squeeze(-1) / denom
            results.append(masked_score.unsqueeze(1))

        return torch.cat(results, dim=1).unsqueeze(1)

    def _boolean(self, x, edge_index_seq, edge_weight_seq):
        phi1_val = self.phi1._quantitative(x).squeeze(1) > 0.5
        phi2_val = self.phi2._quantitative(x).squeeze(1) > 0.5
        B, N, T = phi1_val.shape

        results = []
        for t in range(T):
            edge_index = edge_index_seq[t]
            edge_weights = edge_weight_seq[t]
            edge_dists = self.distance_fn(edge_weights)

            dist = torch.full((N, N), float("inf"), device=x.device)
            dist.fill_diagonal_(0)
            src, dst = edge_index
            dist[src, dst] = edge_dists

            for _ in range(self.max_iter):
                for i in range(N):
                    dist[i] = torch.minimum(dist[i], (dist[i].unsqueeze(0) + dist).min(dim=0).values)

            dist_mask = (dist >= self.d1) & (dist <= self.d2)
            phi1_t = phi1_val[:, :, t]
            phi2_t = phi2_val[:, :, t]

            score = phi2_t.clone()
            for _ in range(self.max_iter):
                phi1_mask = phi1_t.unsqueeze(2) & phi1_t.unsqueeze(1)
                adj = (dist < float("inf")).to(torch.bool)
                batch_adj = adj.unsqueeze(0).expand(B, -1, -1) & phi1_mask
                score = score | (torch.bmm(batch_adj.float(), score.unsqueeze(2).float()).squeeze(2) > 0)

            final = torch.bmm(dist_mask[None].float().expand(B, -1, -1), score.unsqueeze(2).float()).squeeze(2) > 0
            results.append(final.unsqueeze(1))

        return torch.cat(results, dim=1).unsqueeze(1).float()

class DynamicEscape(Node):
    def __init__(
        self,
        phi: Node,
        distance_fn: Callable[[Tensor], Tensor],
        d1: float,
        d2: Optional[float] = None,
        beta: float = 10.0,
        max_iter: int = 10,
    ):
        super().__init__()
        self.phi = phi
        self.distance_fn = distance_fn
        self.d1 = d1
        self.d2 = float('inf') if d2 is None else d2
        self.beta = beta
        self.max_iter = max_iter

    def __str__(self):
        return f"DynamicEscape({self.phi}, d1={self.d1}, d2={self.d2})"

    def time_depth(self):
        return self.phi.time_depth()

    def _soft_distance_matrix(self, edge_index, edge_dists, N):
        device = edge_dists.device
        dist = torch.full((N, N), float("inf"), device=device)
        dist.fill_diagonal_(0)
        src, dst = edge_index
        dist[src, dst] = edge_dists

        for _ in range(self.max_iter):
            current = dist.unsqueeze(0) + dist.unsqueeze(1)
            softmin = -torch.logsumexp(-self.beta * current, dim=2) / self.beta
            dist = torch.minimum(dist, softmin)
        return dist

    def _quantitative(self, x, edge_index_seq, edge_weight_seq, normalize=False):
        phi_val = self.phi._quantitative(x, normalize).squeeze(1)
        B, N, T = phi_val.shape

        results = []
        for t in range(T):
            edge_index = edge_index_seq[t]
            edge_weights = edge_weight_seq[t]
            edge_dists = self.distance_fn(edge_weights)
            soft_dist = self._soft_distance_matrix(edge_index, edge_dists, N)

            mask1 = torch.sigmoid(self.beta * (soft_dist - self.d1))
            mask2 = torch.sigmoid(self.beta * (soft_dist - self.d2))
            dist_mask = mask1 * (1 - mask2)

            phi_t = phi_val[:, :, t]
            dist_mask_batch = dist_mask[None].expand(B, -1, -1)
            denom = dist_mask.sum(1, keepdim=True).clamp(min=1e-6)
            masked_score = torch.bmm(dist_mask_batch, phi_t.unsqueeze(-1)).squeeze(-1) / denom
            results.append(masked_score.unsqueeze(1))

        return torch.cat(results, dim=1).unsqueeze(1)

    def _boolean(self, x, edge_index_seq, edge_weight_seq):
        phi_val = self.phi._quantitative(x).squeeze(1) > 0.5
        B, N, T = phi_val.shape

        results = []
        for t in range(T):
            edge_index = edge_index_seq[t]
            edge_weights = edge_weight_seq[t]
            edge_dists = self.distance_fn(edge_weights)

            dist = torch.full((N, N), float("inf"), device=x.device)
            dist.fill_diagonal_(0)
            src, dst = edge_index
            dist[src, dst] = edge_dists

            for _ in range(self.max_iter):
                for i in range(N):
                    dist[i] = torch.minimum(dist[i], (dist[i].unsqueeze(0) + dist).min(dim=0).values)

            dist_mask = (dist >= self.d1) & (dist <= self.d2)
            phi_t = phi_val[:, :, t]
            score = torch.bmm(dist_mask[None].float().expand(B, -1, -1), phi_t.unsqueeze(2).float()).squeeze(2) > 0
            results.append(score.unsqueeze(1))

        return torch.cat(results, dim=1).unsqueeze(1).float()

'''
from collections import deque # for BFS, controlla che vada bene con la grafo comp. di torch
# usa pytorch geometric
from typing import Dict, Set # for visited structure, which tracks which nodes you've already seen at which distances

class Reach(Node):
    
    # REACH: checks whether a location satisfying property ϕ₂ can be reached from the current node, through a path 
    #        of nodes where each node along the way (excluding the destination) satisfies ϕ₁, and the distance of 
    #        the path lies in the interval [d₁, d₂]
    
    # Intuition: From a given location, is there a path through nodes satisfying ϕ₁ that leads to a node satisfying ϕ₂, 
    #            such that the total distance of the path is within the interval [d₁, d₂]?
    
    def __init__(self, phi1: Node, phi2: Node, distance_matrix: Tensor,
                 d_min: float = 0.0, d_max: float = 1.0,
                 unbound: bool = False, right_unbound: bool = False):
        super().__init__()
        self.phi1 = phi1 # logical expressions to monitor
        self.phi2 = phi2
        self.distance_matrix = distance_matrix
        self.d_min = d_min # Bounds for the distance interval [d₁, d₂] — the allowed range for reachability
        self.d_max = d_max
        self.unbound = unbound # disables all distance constraints with infinite range
        self.right_unbound = right_unbound # models [d₁, ∞]

    def __str__(self):
        if self.unbound:
            return f"reach ( {self.phi1} => {self.phi2} )"
        right = "inf" if self.right_unbound else str(self.d_max)
        return f"reach_[{self.d_min}, {right}]({self.phi1}, {self.phi2})"

    def time_depth(self):
        return max(self.phi1.time_depth(), self.phi2.time_depth())

    def _to_B_T_N(self, z: Tensor, N: int) -> Tensor:
        """Ensure tensor is shaped (B, T, N) regardless of input shape"""
        # riorganizza i tensori (forma (B, C, T) → (B, T, N))
        if z.ndim == 3:
            B, C, T = z.shape
            if C == 1: # caso in cui c'è la stessa formula per tutti i nodi
                return z.squeeze(1).unsqueeze(-1).repeat(1, T, N) # (B, T, N)
            elif C == N: # caso in cui c'è formula diversa per ciascun nodo
                return z.permute(0, 2, 1) # (B, N, T) → (B, T, N)
        raise ValueError(f"Unsupported tensor shape for Reach: {z.shape}")

    def _distance_valid(self, dist: float) -> bool:
        if self.unbound:
            return True
        elif self.right_unbound:
            return dist >= self.d_min
        else:
            return self.d_min <= dist <= self.d_max

    def _boolean(self, x: Tensor) -> Tensor:
        B, N, T = x.shape # B, T, N
        D = self.distance_matrix
        
        # I'm converting the boolean output of phi1 and phi2 to the shape (B, T, N), where
        # B = batch size, T = time steps, N = number of nodes
        z1 = self._to_B_T_N(self.phi1._boolean(x), N)
        z2 = self._to_B_T_N(self.phi2._boolean(x), N)
        
        # Initialize an output tensor to False (default: ϕ₁ R ϕ₂ is false until proven otherwise)
        output = torch.zeros((B, T, N), dtype=torch.bool)

        for b in range(B):
            for t in range(T):
                for src in range(N):
                    visited: Dict[int, Set[float]] = {i: set() for i in range(N)} # Prevent revisiting a node at the same cumulative distance (STREL says the path must not revisit a node at the same cost)
                    queue = deque([(src, 0.0)]) # Start a breadth-first search (BFS) from node src with distance 0

                    while queue:
                        node, dist = queue.popleft()

                        # If you reach a different node (node ≠ src) and it satisfies ϕ₂, and the path distance is valid, then ϕ₁ R[d₁,d₂] ϕ₂ is True at src
                        if node != src and self._distance_valid(dist) and z2[b, t, node]:
                            output[b, t, src] = True
                            break

                        for neighbor in range(N):
                            step = D[node, neighbor].item()
                            new_dist = dist + step
                            if step > 0 and (self.unbound or new_dist <= self.d_max):
                                if neighbor == src or z1[b, t, neighbor]: # This ensures traversal continues only through nodes satisfying ϕ₁
                                    if new_dist not in visited[neighbor]:
                                        visited[neighbor].add(new_dist)
                                        queue.append((neighbor, new_dist))

        return output.unsqueeze(1)

    def _quantitative(self, x: Tensor, normalize=False) -> Tensor:
        B, N, T = x.shape
        D = self.distance_matrix

        z1 = self._to_B_T_N(self.phi1._quantitative(x, normalize), N)
        z2 = self._to_B_T_N(self.phi2._quantitative(x, normalize), N)
        output = torch.full((B, T, N), float('-inf'))

        for b in range(B):
            for t in range(T):
                for src in range(N):
                    visited: Dict[int, Set[float]] = {i: set() for i in range(N)}
                    queue = deque([(src, 0.0, float('inf'))]) # In addition to node and distance, you track rob: robustness value (initialized to ∞)

                    while queue:
                        node, dist, rob = queue.popleft()

                        if node != src and self._distance_valid(dist):
                            combined = min(rob, z2[b, t, node]) # At each step, update rob
                            output[b, t, src] = max(output[b, t, src], combined) # Final robustness is the maximum robustness across all valid paths (You apply ⊕ (choose) operator: pick the path with the greatest robustness score. This is Table 1, line 6)

                        for neighbor in range(N):
                            step = D[node, neighbor].item()
                            new_dist = dist + step
                            if step > 0 and (self.unbound or new_dist <= self.d_max):
                                if neighbor == src or z1[b, t, neighbor] > float('-inf'):
                                    if new_dist not in visited[neighbor]:
                                        visited[neighbor].add(new_dist)
                                        new_rob = min(rob, z1[b, t, neighbor])
                                        queue.append((neighbor, new_dist, new_rob))

        return output.unsqueeze(1)


class Escape(Node):
    
    # ESCAPE: checks if it's possible to leave the current region by following a path composed only of nodes 
    #         satisfying ϕ, and ending at a node at a certain distance away, specifically between [d₁, d₂]
    
    # Intuition: From a location, is it possible to escape via a route where all intermediate nodes satisfy ϕ, 
    #            and the distance between the start and end node is within [d₁, d₂]?
    
    def __init__(self, phi: Node, distance_matrix: Tensor,
                 d_min: float = 0.0, d_max: float = 1.0,
                 unbound: bool = False, right_unbound: bool = False):
        super().__init__()
        self.phi = phi
        self.distance_matrix = distance_matrix
        self.d_min = d_min
        self.d_max = d_max
        self.unbound = unbound
        self.right_unbound = right_unbound

    def __str__(self):
        if self.unbound:
            return f"escape ( {self.phi} )"
        right = "inf" if self.right_unbound else str(self.d_max)
        return f"escape_[{self.d_min}, {right}]({self.phi})"

    def time_depth(self):
        return self.phi.time_depth()

    def _to_B_T_N(self, z: Tensor, N: int) -> Tensor:
        if z.ndim == 3:
            B, C, T = z.shape
            if C == 1:
                return z.squeeze(1).unsqueeze(-1).repeat(1, T, N)
            elif C == N:
                return z.permute(0, 2, 1)
        raise ValueError(f"Unsupported tensor shape for Escape: {z.shape}")

    def _distance_valid(self, dist: float) -> bool:
        if self.unbound:
            return True
        elif self.right_unbound:
            return dist >= self.d_min
        else:
            return self.d_min <= dist <= self.d_max

    def _boolean(self, x: Tensor) -> Tensor:
        B, N, T = x.shape
        D = self.distance_matrix

        z = self._to_B_T_N(self.phi._boolean(x), N)
        output = torch.zeros((B, T, N), dtype=torch.bool)

        # Per ogni batch b, tempo t, nodo sorgente src:
        for b in range(B):
            for t in range(T):
                for src in range(N):
                    
                    # Per ogni src esamina tutti i nodi dst, se: 
                    #   - distanza ∈ [d_min, d_max]
                    #   - dst soddisfa phi
                    # Allora Escape è True per src (a tempo t)
                    # In sintesi: può fuggire da qualche parte dove phi è vera?
                    
                    for dst in range(N):
                        dist = D[src, dst].item()
                        if self._distance_valid(dist) and z[b, t, dst]:
                            output[b, t, src] = True
                            break

        return output.unsqueeze(1)

    def _quantitative(self, x: Tensor, normalize=False) -> Tensor:
        B, N, T = x.shape
        D = self.distance_matrix

        z = self._to_B_T_N(self.phi._quantitative(x, normalize), N)
        output = torch.full((B, T, N), float('-inf'))

        for b in range(B):
            for t in range(T):
                for src in range(N):
                    for dst in range(N):
                        dist = D[src, dst].item()
                        if self._distance_valid(dist):
                            output[b, t, src] = max(output[b, t, src], z[b, t, dst]) # Similar to reach, but doesn’t accumulate robustness along a path
                            #Just checks the robustness value of the destination node, assuming ϕ must hold at that destination

        return output.unsqueeze(1)
'''  