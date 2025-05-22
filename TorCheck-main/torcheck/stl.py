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


from torch import nn
from typing import Callable, Optional

class Reach(Node):
# The Reach class checks whether a node can reach another node that 
# satisfies a certain property (phi2) through a path of nodes satisfying 
# another property (phi1), within a specified distance interval [d1, d2]
    def __init__(
        self,
        phi1: Node,
        phi2: Node,
        distance_fn: Callable[[Tensor], Tensor], # distance_fn is expected to be: a function that takes one argument of type Tensor and returns a value of type Tensor
        edge_index: Tensor,
        edge_weights: Tensor,
        d1: float,
        d2: Optional[float] = None,
        num_nodes: Optional[int] = None,
        beta: float = 10.0,
        max_iter: int = 10,
    ):
        super().__init__()
        self.phi1 = phi1
        self.phi2 = phi2
        self.distance_fn = distance_fn # maps weights to distances
        self.edge_index = edge_index # defines the graph: a 2×E (standard format in PyTorch Geometric, E is the number of edges) tensor of source/target indices
        self.edge_weights = edge_weights # edge-wise distances (1D tensor of size E) used in bounding the reach (line 5 in alg 4)
        self.edge_dists = self.distance_fn(edge_weights)  # cache distances, used for computing things like: shortest paths (softmin), distance masks (for [d₁, d₂] bounds), soft adjacency matrices
        self.d1 = d1
        self.d2 = float('inf') if d2 is None else d2 # If d2 is infinite, the operator corresponds to the unbounded case in Algorithm 4
        self.num_nodes = num_nodes or int(edge_index.max().item()) + 1 # num_nodes is inferred if not provided
        self.beta = beta # controls the softness of approximations (sigmoid, softmin)
        self.max_iter = max_iter # limits iterations of propagation — replaces while loop in Algorithm 4

    def __str__(self) -> str:
        return f"Reach({self.phi1}, {self.phi2}, d1={self.d1}, d2={self.d2})"

    def time_depth(self) -> int:
        return max(self.phi1.time_depth(), self.phi2.time_depth())

    def _build_soft_adjacency(self, phi1_val: Tensor) -> Tensor:
        # Builds a differentiable adjacency matrix that only allows propagation 
        # through φ₁-satisfying nodes (lines 7–10 of Algorithm 3)
        src, dst = self.edge_index # nodi di partenza e di destinazione nella matrice edge_index
        A = torch.zeros(self.num_nodes, self.num_nodes, device=phi1_val.device) # A contains softened weights on valid edges, where smaller distance ⇒ higher weight
        A[src, dst] = torch.exp(-self.beta * self.edge_dists).clamp(min=1e-6) # clamp(min=1e-6) evita zeri che potrebbero causare problemi numerici
        # A è la matrice di adiacenza pesata: dimensione [N, N], inizialmente tutta a 0.
        # Per ogni arco (i, j), assegna un valore decrescente in base alla distanza: a[i, j]=exp(-beta * distance)
        # più corta la distanza, più grande il peso (cioè più facilmente attraversabile)
        
        # Uses φ₁ to gate the edges (both src and dst must satisfy φ₁): 
        # mimics the if z1[...] condition in Algorithm 3, line 10
        phi1_gate = torch.sigmoid(self.beta * phi1_val) # Applica la sigmoide a ogni valore: se φ₁ è “attivata” in un nodo (positiva), il valore sarà vicino a 1, β rende la sigmoide più netta (con β → ∞, tende a 0/1 hard gating)
        gate_mask = phi1_gate.unsqueeze(1) * phi1_gate.unsqueeze(2) # Simula: "permetti attraversamento da i a j solo se sia i che j soddisfano φ₁"
        # dimensione gate_mask è [B, N, N] perchè: phi1_gate diventa [B, 1, N] e [B, N, 1], e il loro prodotto fa una matrice [B, N, N]
        # phi1_val ha dimensione [B, N, T] o [B, N] (a seconda che ci sia tempo fissato)
        
        return A.unsqueeze(0) * gate_mask 
        # A ha shape [N, N], quindi unsqueeze(0) → [1, N, N]
        # gate_mask ha shape [B, N, N]
        # Output: [B, N, N] matrice di adiacenza soft per ogni batch

        # La funzione ritorna una matrice di adiacenza soft (differenziabile) che rappresenta il grafo, ma:
        # - i pesi sono “soft” (esponenziali decrescenti con la distanza),
        # - le connessioni sono attivate solo se entrambi i nodi coinvolti soddisfano φ₁
        
    def _soft_distance_matrix(self) -> Tensor: # calcola una matrice delle distanze minime tra tutti i nodi del grafo, in modo differenziabile
        
        # Implements distance propagation using softmin, analogous to 
        # Dijkstra or Floyd-Warshall. This is the differentiable 
        # equivalent of computing d′ = d +_B f(w) in Algorithm 3, line 9.
        N = self.num_nodes # numero di nodi nel grafo
        device = self.edge_weights.device # assicura che tutto venga eseguito su CPU o GPU coerentemente
        
        # Initialize the shortest-path distance matrix.
        dist = torch.full((N, N), float("inf"), device=device) # Crea una matrice N x N piena di ∞ (infinito): ogni nodo è inizialmente irraggiungibile da ogni altro
        dist.fill_diagonal_(0) # Imposta 0 sulla diagonale: la distanza da un nodo a sé stesso è 0
        src, dst = self.edge_index # self.edge_index = coppie (sorgente, destinazione) degli archi
        dist[src, dst] = self.edge_dists # self.edge_dists = distanza (già calcolata da distance_fn) su ogni arco
        # Imposta le distanze note nei corrispondenti elementi della matrice

        #Uses log-sum-exp softmin over paths: approximates min-distance for all pairs.
        for _ in range(self.max_iter): # stiamo costruendo tutte le combinazioni di cammini a due passi
        # Il ciclo for _ in range(self.max_iter) simula un algoritmo di tipo Floyd–Warshall, ma solo per 
        # un numero finito di iterazioni (per motivi computazionali). Ogni iterazione permette la propagazione 
        # della distanza su percorsi sempre più lunghi    
        
            current = dist.unsqueeze(0) + dist.unsqueeze(1)
            # dist.unsqueeze(0) → shape [1, N, N] (prima matrice)
            # dist.unsqueeze(1) → shape [N, 1, N] (seconda matrice)
            # sommandoli otteniamo current[i, j, k] = dist[i, j] + dist[j, k]
            # → rappresenta il costo del cammino i → j → k
            
            softmin = -torch.logsumexp(-self.beta * current, dim=2) / self.beta
            # Applica una softmin sull’ultima dimensione (tutti i j intermedi):
                # logsumexp è un modo differenziabile di approssimare il minimo.
                # Più β è grande, più il risultato si avvicina al vero minimo.
                # Restituisce la distanza più breve (soft) da i → k passando per un nodo intermedio
            
            dist = torch.minimum(dist, softmin) # Aggiorna dist[i, k] solo se il nuovo percorso trovato tramite softmin è più corto
            
        return dist # Restituisce una matrice dist[i, j] che rappresenta la distanza più breve (approssimata) tra ogni coppia di nodi i → j, usando un'operazione softmin che può essere derivata

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        # Evaluates robustness of φ₁ and φ₂ at each node and time step
        
        # for each node and time step, propagates satisfaction of phi2 
        # through phi1-valid paths and accumulates over paths whose 
        # distances lie within [d1, d2]
        
        # Per ogni nodo e istante temporale:
            # Valuta φ₂ nei nodi destinazione.
            # Propaga il valore di φ₂ solo lungo nodi che soddisfano φ₁.
            # Tiene conto solo dei cammini la cui lunghezza soft è compresa tra d₁ e d₂.
            # Restituisce quanto fortemente ogni nodo soddisfa l'operatore φ₁ R[d₁,d₂] φ₂
        
        # Calcola la robustezza quantitativa delle formule φ₁ e φ₂ per ogni nodo e istante temporale
        phi1_val = self.phi1._quantitative(x, normalize).squeeze(1)
        phi2_val = self.phi2._quantitative(x, normalize).squeeze(1)
        B, N, T = phi1_val.shape 
        # Output: tensori di forma [B, N, T] dove: 
        # B = batch size (numero di esempi), 
        # N = numero di nodi,
        # T = numero di istanti temporali (time steps)

        # Builds a soft mask for the interval [d₁, d₂]. This replaces the 
        # if d1 ≤ d ≤ d2 condition in Algorithm 3, line 10.
        soft_dist = self._soft_distance_matrix() # soft_dist[i, j]: distanza soft da nodo i a j
        mask1 = torch.sigmoid(self.beta * (soft_dist - self.d1)) # mask1 attiva i valori dove dist ≥ d1
        mask2 = torch.sigmoid(self.beta * (soft_dist - self.d2)) # mask2 attiva i valori dove dist ≥ d2
        dist_mask = mask1 * (1 - mask2) # La differenza mask1 * (1 - mask2) approssima 1 se d ∈ [d1, d2], altrimenti 0
        # dist_mask ha forma [N, N]
        
        results = []
        # Iterative propagation of robustness values from φ₂ nodes over 
        # the φ₁-gated graph — matches the BFS-style loop of Algorithm 4, lines 9–21
        for t in range(T):
            phi1_t = phi1_val[:, :, t] # Estrai le valutazioni di φ₁ e φ₂ al tempo t
            phi2_t = phi2_val[:, :, t]
            soft_adj = self._build_soft_adjacency(phi1_t) # Rende disponibili solo i cammini dove entrambi i nodi soddisfano φ₁
            # soft_adj ha forma [B, N, N]
            
            score = phi2_t.clone()
            for _ in range(self.max_iter): # Propaga la robustezza φ₂ attraverso i cammini φ₁-validi
                score = torch.maximum(score, torch.bmm(soft_adj, score.unsqueeze(-1)).squeeze(-1)) # score raccoglie il massimo valore raggiungibile da ogni nodo

            dist_mask_batch = dist_mask[None].expand(B, -1, -1) # Applica dist_mask come filtro ai valori di score, cioè solo i nodi raggiungibili entro [d₁, d₂] contribuiscono
            denom = dist_mask.sum(1, keepdim=True).clamp(min=1e-6) # Fa una media pesata, normalizzando per evitare divisioni per zero (clamp)
            masked_score = torch.bmm(dist_mask_batch, score.unsqueeze(-1)).squeeze(-1) / denom
            # masked score: Aggregates scores over valid destinations. Implements 
            # the ⊕ accumulation of all (ℓ, ℓ′) with d ∈ [d₁, d₂] as in Algorithm 3, lines 22–24
            results.append(masked_score.unsqueeze(1)) # Salva il risultato per il tempo t

        return torch.cat(results, dim=1).unsqueeze(1) # Concatena i risultati lungo l’asse temporale → forma finale: [B, 1, T]

    def _boolean(self, x: Tensor) -> Tensor:
        # evaluates the same reachability condition with strict true/false logic
        
        # Per ogni nodo e istante temporale t, verifica se esiste un nodo che:
            # Soddisfa φ₂.
            # È raggiungibile attraverso nodi che soddisfano φ₁.
            # Si trova a distanza compresa tra d₁ e d₂

        phi1_val = self.phi1._quantitative(x).squeeze(1) > 0.5 # Se la robustezza > 0.5, allora si considera vera, altrimenti falsa
        phi2_val = self.phi2._quantitative(x).squeeze(1) > 0.5
        B, N, T = phi1_val.shape # phi1_val, phi2_val → shape [B, N, T]

        # Costruisce la matrice di adiacenza del grafo adj[i, j] = True se esiste arco i → j
        src, dst = self.edge_index
        adj = torch.zeros(N, N, dtype=torch.bool, device=x.device)
        adj[src, dst] = True

        # Distanze Minime (tipo Floyd-Warshall): calcola le distanze minime tra i nodi
        dist = torch.full((N, N), float("inf"), device=x.device) # Inizializza matrice delle distanze
        dist.fill_diagonal_(0) # Imposta a 0 la diagonale (distanza da sé stessi)
        dist[src, dst] = self.edge_dists # Inserisce le distanze reali sugli archi

        # Calcolo delle distanze minime tra tutti i nodi: Approssima l'algoritmo di Floyd–Warshall per trovare distanze minime tra tutti i nodi
        for _ in range(self.max_iter):
            for i in range(N):
                combined = dist[i].unsqueeze(0) + dist
                dist[i] = torch.minimum(dist[i], combined.min(dim=0).values) # dist[i, j]: distanza più breve da nodo i a nodo j

        dist_mask = (dist >= self.d1) & (dist <= self.d2) # dist_mask[i, j] = True se la distanza tra i e j è dentro l’intervallo

        results = []
        for t in range(T): # per ogni istante di tempo estrae una valutazione delle fo0rmule
            phi1_t = phi1_val[:, :, t]
            phi2_t = phi2_val[:, :, t]

            # Emulates line 15 in Algorithm 3, where previous values are 
            # combined using ⊕ (boolean OR)
            score = phi2_t.clone() # Inizialmente score[i] = True se il nodo i soddisfa φ₂
            for _ in range(self.max_iter):
                phi1_mask = phi1_t.unsqueeze(2) & phi1_t.unsqueeze(1) # Costruisce phi1_mask per permettere solo la propagazione tra nodi che soddisfano φ₁
                batch_adj = adj.unsqueeze(0).expand(B, -1, -1) & phi1_mask
                propagated = torch.bmm(batch_adj.float(), score.unsqueeze(2).float()).squeeze(2) > 0 # Applica la propagazione: un nodo eredita True se almeno uno dei suoi vicini φ₁-validi ha True
                score = score | propagated

            # Final result checks if there's any reachable φ₂ node in the valid distance interval
            final = torch.bmm(dist_mask[None].float().expand(B, -1, -1), score.unsqueeze(2).float()).squeeze(2) > 0
            # Final moltiplica dist_mask per score: seleziona solo i nodi che:
                # soddisfano φ₂
                # sono raggiungibili entro distanza valida.
            # Se almeno un nodo soddisfa entrambe le condizioni, allora final = True.
            
            results.append(final.unsqueeze(1)) # salva il risultato

        return torch.cat(results, dim=1).unsqueeze(1).float() # Restituisce un tensore di shape [B, 1, T], con 1.0 se il nodo soddisfa la formula φ₁ R[d₁,d₂] φ₂, 0.0 altrimenti


class Escape(Node):
    def __init__(
        self,
        phi: Node,
        distance_fn: Callable[[Tensor], Tensor],
        edge_index: Tensor,
        edge_weights: Tensor,
        d1: float,
        d2: Optional[float] = None,
        num_nodes: Optional[int] = None,
        beta: float = 10.0,
        max_iter: int = 10,
    ):
        super().__init__()
        self.phi = phi
        self.distance_fn = distance_fn
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.edge_dists = self.distance_fn(edge_weights)
        self.d1 = d1
        self.d2 = float('inf') if d2 is None else d2
        self.num_nodes = num_nodes or int(edge_index.max().item()) + 1
        self.beta = beta
        self.max_iter = max_iter

    def __str__(self) -> str:
        return f"Escape({self.phi}, d1={self.d1}, d2={self.d2})"

    def time_depth(self) -> int:
        return self.phi.time_depth()

    def _soft_distance_matrix(self) -> Tensor:
        N = self.num_nodes
        device = self.edge_weights.device
        dist = torch.full((N, N), float("inf"), device=device)
        dist.fill_diagonal_(0)
        src, dst = self.edge_index
        dist[src, dst] = self.edge_dists

        for _ in range(self.max_iter):
            current = dist.unsqueeze(0) + dist.unsqueeze(1)
            softmin = -torch.logsumexp(-self.beta * current, dim=2) / self.beta
            dist = torch.minimum(dist, softmin)
        return dist

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        phi_val = self.phi._quantitative(x, normalize).squeeze(1)
        B, N, T = phi_val.shape

        # Computes soft mask as in lines 23–24 of Algorithm 5: 
        # aggregates φ values at destinations with valid distance
        soft_dist = self._soft_distance_matrix()
        mask1 = torch.sigmoid(self.beta * (soft_dist - self.d1))
        mask2 = torch.sigmoid(self.beta * (soft_dist - self.d2))
        dist_mask = mask1 * (1 - mask2)

        results = []
        # Aggregates destination φ values over valid [d₁, d₂] 
        # (as in s(ℓ) = ⊕ e[ℓ, ℓ′])
        for t in range(T):
            phi_t = phi_val[:, :, t]
            dist_mask_batch = dist_mask[None].expand(B, -1, -1)
            denom = dist_mask.sum(1, keepdim=True).clamp(min=1e-6)
            masked_score = torch.bmm(dist_mask_batch, phi_t.unsqueeze(-1)).squeeze(-1) / denom
            results.append(masked_score.unsqueeze(1))

        return torch.cat(results, dim=1).unsqueeze(1)

    def _boolean(self, x: Tensor) -> Tensor:
        # Checks if any reachable node at distance ∈ [d₁, d₂] satisfies φ — as per line 12 of Algorithm 5
        phi_val = self.phi._quantitative(x).squeeze(1) > 0.5
        B, N, T = phi_val.shape

        src, dst = self.edge_index
        adj = torch.zeros(N, N, dtype=torch.bool, device=x.device)
        adj[src, dst] = True

        dist = torch.full((N, N), float("inf"), device=x.device)
        dist.fill_diagonal_(0)
        dist[src, dst] = self.edge_dists

        for _ in range(self.max_iter):
            for i in range(N):
                combined = dist[i].unsqueeze(0) + dist
                dist[i] = torch.minimum(dist[i], combined.min(dim=0).values)

        dist_mask = (dist >= self.d1) & (dist <= self.d2)

        results = []
        for t in range(T):
            phi_t = phi_val[:, :, t]
            score = torch.bmm(dist_mask[None].float().expand(B, -1, -1), phi_t.unsqueeze(2).float()).squeeze(2) > 0
            results.append(score.unsqueeze(1))

        return torch.cat(results, dim=1).unsqueeze(1).float()