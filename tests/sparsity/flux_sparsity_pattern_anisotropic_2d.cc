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

// @sect3{Include files}

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "../tests.h"



// @sect3{The main() function}
// A 2D test of the sparsity pattern for a locally refined domain.
using namespace dealii;

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
create_flux_pattern(DoFHandler<dim> &dof_handler, DynamicSparsityPattern &dsp)
{
  Table<2, DoFTools::Coupling> cell_coupling(1, 1);
  Table<2, DoFTools::Coupling> face_coupling(1, 1);
  cell_coupling[0][0] = DoFTools::always;
  face_coupling[0][0] = DoFTools::always;

  const AffineConstraints<double> constraints;
  const bool                      keep_constrained_dofs = true;

  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp,
                                       constraints,
                                       keep_constrained_dofs,
                                       cell_coupling,
                                       face_coupling,
                                       numbers::invalid_subdomain_id);
  dsp.print(deallog.get_file_stream());
}


int
main()
{
  initlog();
  const int dim = 2;

  // create a square with refined subdomain
  Triangulation<dim> triangulation;
  make_anisotropic_grid(triangulation);

  deallog << "dealii::DoFHandler" << std::endl;
  {
    // Generate Q1 dofs for the mesh grid
    DoFHandler<dim> dof_handler(triangulation);
    const FE_Q<dim> finite_element(1);
    dof_handler.distribute_dofs(finite_element);

    // Compute the sparsity pattern specifying
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    create_flux_pattern(dof_handler, dsp);
  }
}
