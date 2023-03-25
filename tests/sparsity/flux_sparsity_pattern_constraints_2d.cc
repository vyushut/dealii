/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

//{A 2D test about the flux sparsity pattern for an anisotropic mesh with
// hanging nodes same as in
// sparsity/face_sparsity_pattern_2d_anisotropic.cc. Here, we assemble and
// output the flux matrix without and with constraints.
//}

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include "deal.II/../../tests/tests.h"


using namespace dealii;

template <int dim>
bool
just_visiting(const typename DoFHandler<dim>::active_cell_iterator &cell,
              const unsigned int                                    face_index)
{
  deallog.get_file_stream() << "Visiting the face " << face_index << " on cell "
                            << cell->index() << std::endl;
  return true;
}

template <int dim>
void
make_anisotropic_grid(Triangulation<dim> &triangulation)
{
  GridGenerator::hyper_cube(triangulation, -2.0, 2.0);
  triangulation.begin_active()->set_refine_flag(
    RefinementCase<dim>::cut_axis(0));
  triangulation.execute_coarsening_and_refinement();

  for (auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->center()(0) > 0.0)
        cell->set_refine_flag(RefinementCase<dim>::cut_axis(0));
      else
        cell->set_refine_flag(RefinementCase<dim>::cut_axis(1));
    }
  triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void
create_and_output_flux_pattern(DoFHandler<dim> &          dof_handler,
                               DynamicSparsityPattern &   dsp,
                               AffineConstraints<double> &constraints)
{
  deallog
    << "Exactly 4 faces should be visited for creating the flux sparsity pattern + 4 faces visited during the assembly"
    << std::endl;
  Table<2, DoFTools::Coupling> coupling(1, 1);
  coupling.fill(DoFTools::always);
  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       constraints,
                                       true,
                                       coupling,
                                       coupling,
                                       numbers::invalid_subdomain_id,
                                       [&](const auto &       cell,
                                           const unsigned int face_index) {
                                         return just_visiting<dim>(cell,
                                                                   face_index);
                                       });
}


template <int dim>
void
assembly_and_print(const FiniteElement<dim> & fe,
                   DoFHandler<dim> &          dof_handler,
                   SparsityPattern &          sparsity_pattern,
                   AffineConstraints<double> &constraints,
                   SparseMatrix<double> &     global_matrix)
{
  global_matrix.reinit(sparsity_pattern);

  const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
  const QGauss<dim - 1>                face_quadrature(1 + 1);
  FEInterfaceValues<dim>               fe_interface_values(fe,
                                             face_quadrature,
                                             update_values | update_gradients |
                                               update_JxW_values |
                                               update_normal_vectors);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            continue;
          auto neighbor = cell->neighbor_or_periodic_neighbor(f);
          if (!neighbor->is_active())
            continue;
          if (!cell->neighbor_is_coarser(f) &&
              neighbor->index() > cell->index())
            continue;
          if (!just_visiting<dim>(cell, f))
            continue;
          const unsigned int invalid_subface = numbers::invalid_unsigned_int;
          int                first;
          int                second;
          if (cell->neighbor_is_coarser(f))
            {
              first  = cell->neighbor_of_coarser_neighbor(f).first;
              second = cell->neighbor_of_coarser_neighbor(f).second;
            }
          else
            {
              first  = cell->neighbor_of_neighbor(f);
              second = numbers::invalid_unsigned_int;
            }
          fe_interface_values.reinit(
            cell, f, invalid_subface, neighbor, first, second);
          const unsigned int n_interface_dofs =
            fe_interface_values.n_current_interface_dofs();
          FullMatrix<double> local_stabilization(n_interface_dofs,
                                                 n_interface_dofs);
          for (unsigned int q = 0; q < fe_interface_values.n_quadrature_points;
               ++q)
            {
              const Tensor<1, dim> normal = fe_interface_values.normal(q);
              for (unsigned int i = 0; i < n_interface_dofs; ++i)
                for (unsigned int j = 0; j < n_interface_dofs; ++j)
                  {
                    local_stabilization(i, j) +=
                      (24) * normal *
                      fe_interface_values.jump_in_shape_gradients(i, q) *
                      normal *
                      fe_interface_values.jump_in_shape_gradients(j, q) *
                      fe_interface_values.JxW(q);
                  }
            }
          const std::vector<types::global_dof_index>
            local_interface_dof_indices =
              fe_interface_values.get_interface_dof_indices();

          constraints.distribute_local_to_global(local_stabilization,
                                                 local_interface_dof_indices,
                                                 global_matrix);
        }
    }
}

template <int dim, bool constrained>
void
check()
{
  Triangulation<dim> triangulation;
  make_anisotropic_grid(triangulation);

  DoFHandler<dim> dof_handler(triangulation);
  const FE_Q<dim> finite_element(1);
  dof_handler.distribute_dofs(finite_element);

  AffineConstraints<double> constraints;
  if (constrained)
    {
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
    }

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  create_and_output_flux_pattern(dof_handler, dsp, constraints);

  SparseMatrix<double> global_matrix;
  SparsityPattern      sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  assembly_and_print<dim>(
    finite_element, dof_handler, sparsity_pattern, constraints, global_matrix);
  global_matrix.print_formatted(deallog.get_file_stream(), 0, false);
}

int
main()
{
  initlog();
  deallog.push("2d::unconstrained");
  check<2, false>();
  deallog.pop();
  deallog.push("2d::constrained");
  check<2, true>();
  deallog.pop();
}
