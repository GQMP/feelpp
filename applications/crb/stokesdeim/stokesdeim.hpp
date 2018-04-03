#ifndef FEELPP_STOKES_HPP
#define FEELPP_STOKES_HPP 1

#include <feel/options.hpp>
#include <feel/feelcrb/modelcrbbase.hpp>
#include <feel/feelcrb/crbsaddlepoint.hpp>
#include <feel/feelcrb/crbmodelsaddlepoint.hpp>



namespace Feel
{

FEELPP_EXPORT AboutData
    makeStokesDeimAbout( std::string const& str = "stokesdeim" );

struct FEELPP_EXPORT StokesDeimConfig
{
    typedef Mesh<Simplex<2>> mesh_type;
    typedef THch_type<1,mesh_type> space_type;
};

class FEELPP_EXPORT StokesDeim :
    public ModelCrbBase<ParameterSpace<>, StokesDeimConfig::space_type, UseBlock >
{
    typedef StokesDeim self_type;
    typedef ModelCrbBase<ParameterSpace<>, StokesDeimConfig::space_type, UseBlock > super_type;

 public :
    typedef typename StokesDeimConfig::space_type space_type;
    typedef boost::tuple<beta_vector_type,  std::vector<beta_vector_type> > beta_type;
    typedef typename super_type::sparse_matrix_ptrtype sparse_matrix_ptrtype;
    using super_type::computeBetaQm;

    StokesDeim();
    void initModel() override;
    beta_type computeBetaQm( parameter_type const& mu ) override;
    value_type output( int output_index, parameter_type const& mu , element_type& u, bool need_to_solve=false) override;
    sparse_matrix_ptrtype assembleForMDEIM( parameter_type const& mu, int const& tag ) override;
};


} // namespace Feel

#endif
