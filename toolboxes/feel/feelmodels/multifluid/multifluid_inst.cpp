
#include "multifluidconfig.h"

#include <feel/feelmodels/multifluid/multifluid.cpp>

#include <feel/feelmodels/multifluid/helfrichforcemodel.hpp>
#include <feel/feelmodels/multifluid/inextensibilityforcemodel.hpp>
#include <feel/feelmodels/multifluid/linearelasticforcemodel.hpp>
#include <feel/feelmodels/multifluid/skalakforcemodel.hpp>

namespace Feel {
namespace FeelModels {

//template class MultiFluid< FLUIDMECHANICS_CLASS_TYPE, LEVELSET_CLASS_TYPE >;
template class MULTIFLUID_CLASS_INSTANTIATION;

// Register interface forces models
const bool helfrich_interfaceforcesmodel = 
    MULTIFLUID_CLASS_INSTANTIATION::interfaceforces_factory_type::instance().registerProduct( 
            "helfrich", 
            &detail::createInterfaceForcesModel<HelfrichForceModel, typename MULTIFLUID_CLASS_INSTANTIATION::levelset_type> );

const bool inextensibility_interfaceforcesmodel = 
    MULTIFLUID_CLASS_INSTANTIATION::interfaceforces_factory_type::instance().registerProduct( 
            "inextensibility-force", 
            &detail::createInterfaceForcesModel<InextensibilityForceModel, typename MULTIFLUID_CLASS_INSTANTIATION::levelset_type> );

const bool linearelastic_interfaceforcesmodel = 
    MULTIFLUID_CLASS_INSTANTIATION::interfaceforces_factory_type::instance().registerProduct( 
            "linear-elastic-force", 
            &detail::createInterfaceForcesModel<LinearElasticForceModel, typename MULTIFLUID_CLASS_INSTANTIATION::levelset_type> );

const bool skalak_interfaceforcesmodel = 
    MULTIFLUID_CLASS_INSTANTIATION::interfaceforces_factory_type::instance().registerProduct( 
            "skalak-force", 
            &detail::createInterfaceForcesModel<SkalakForceModel, typename MULTIFLUID_CLASS_INSTANTIATION::levelset_type> );

}
}

