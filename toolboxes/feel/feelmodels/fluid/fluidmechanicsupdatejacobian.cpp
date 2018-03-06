/* -*- mode: c++; coding: utf-8; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; show-trailing-whitespace: t -*- vim:fenc=utf-8:ft=cpp:et:sw=4:ts=4:sts=4 
 */

#include <feel/feelmodels/fluid/fluidmechanics.hpp>

#include <feel/feelvf/vf.hpp>

#include <feel/feelmodels/modelvf/fluidmecconvection.hpp>
#include <feel/feelmodels/modelvf/fluidmecstresstensor.hpp>

namespace Feel
{
namespace FeelModels
{

FLUIDMECHANICS_CLASS_TEMPLATE_DECLARATIONS
void
FLUIDMECHANICS_CLASS_TEMPLATE_TYPE::updateJacobian( DataUpdateJacobian & data ) const
{
    using namespace Feel::vf;

    const vector_ptrtype& XVec = data.currentSolution();
    sparse_matrix_ptrtype& J = data.jacobian();
    vector_ptrtype& RBis = data.vectorUsedInStrongDirichlet();
    bool _BuildCstPart = data.buildCstPart();
    bool _doBCStrongDirichlet = data.doBCStrongDirichlet();

    std::string sc=(_BuildCstPart)?" (build cst part)":" (build non cst part)";
    if (this->verbose()) Feel::FeelModels::Log("--------------------------------------------------\n",
                                               this->prefix()+".FluidMechanics","updateJacobian", "start"+sc,
                                               this->worldComm(),this->verboseAllProc());
    boost::mpi::timer thetimer;

    bool BuildNonCstPart = !_BuildCstPart;
    bool BuildCstPart = _BuildCstPart;

    //bool BuildNonCstPart_robinFSI = BuildNonCstPart;
    //if (this->useFSISemiImplicitScheme()) BuildNonCstPart_robinFSI=BuildCstPart;

    //--------------------------------------------------------------------------------------------------//

    auto mesh = this->mesh();
    auto Xh = this->functionSpace();

    size_type rowStartInMatrix = this->rowStartInMatrix();
    size_type colStartInMatrix = this->colStartInMatrix();
    size_type rowStartInVector = this->rowStartInVector();
    auto bilinearForm_PatternDefault = form2( _test=Xh,_trial=Xh,_matrix=J,
                                              _pattern=size_type(Pattern::DEFAULT),
                                              _rowstart=rowStartInMatrix,
                                              _colstart=colStartInMatrix );
    auto bilinearForm_PatternCoupled = form2( _test=Xh,_trial=Xh,_matrix=J,
                                              _pattern=size_type(Pattern::COUPLED),
                                              _rowstart=rowStartInMatrix,
                                              _colstart=colStartInMatrix );

    auto U = Xh->element(XVec, rowStartInVector);
    auto u = U.template element<0>();
    auto v = U.template element<0>();
    auto p = U.template element<1>();
    auto q = U.template element<1>();

    //--------------------------------------------------------------------------------------------------//

    // identity Matrix
    auto const Id = eye<nDim,nDim>();
    // strain tensor
    auto const deft = sym(gradt(u));
    // dynamic viscosity
    auto const& mu = this->densityViscosityModel()->fieldMu();
    auto const& rho = this->densityViscosityModel()->fieldRho();
    // stress tensor
    auto const Sigmat = -idt(p)*Id + 2*idv(mu)*deft;
    //--------------------------------------------------------------------------------------------------//
    //--------------------------------------------------------------------------------------------------//
    //--------------------------------------------------------------------------------------------------//

    boost::mpi::timer timerAssemble;

    //--------------------------------------------------------------------------------------------------//
    // convection terms
    if ( BuildNonCstPart )
    {
        if (this->doStabConvectionEnergy())
        {
            // convection term + stabilisation energy of convection with neumann bc (until outflow bc) ( see Nobile thesis)
            // auto const convecTerm = (trans(val(gradv(u)*idv(*M_P0Rho))*idt(u)) + trans(gradt(u)*val(idv(u)*idv(*M_P0Rho)) ) )*id(v);
            // stabTerm = trans(divt(u)*val(0.5*idv(*M_P0Rho)*idv(u))+val(0.5*idv(*M_P0Rho)*divv(u))*idt(u))*id(v)

            auto const convecTerm = Feel::vf::FeelModels::fluidMecConvectionJacobianWithEnergyStab(u,rho);
            bilinearForm_PatternCoupled +=
                //bilinearForm_PatternDefault +=
                integrate ( _range=M_rangeMeshElements,
                            _expr=convecTerm,
                            _geomap=this->geomap() );
        }
        else
        {
#if 0
            auto const convecTerm = (trans(val(gradv(u)*idv(rho))*idt(u)) + trans(gradt(u)*val(idv(u)*idv(rho)) ) )*id(v);
#else
            auto const convecTerm = Feel::vf::FeelModels::fluidMecConvectionJacobian(u,rho);
#endif
            bilinearForm_PatternCoupled +=
                //bilinearForm_PatternDefault +=
                integrate ( _range=M_rangeMeshElements,
                            _expr=convecTerm,
                            _geomap=this->geomap() );
        }
    }

#if defined( FEELPP_MODELS_HAS_MESHALE )
    if (this->isMoveDomain() && BuildCstPart )
    {
        bilinearForm_PatternCoupled +=
            //bilinearForm_PatternDefault +=
            integrate (_range=M_rangeMeshElements,
                       _expr= -trans(gradt(u)*idv(rho)*idv( this->meshVelocity() ))*id(v),
                       _geomap=this->geomap() );
    }
#endif

    double timeElapsed=timerAssemble.elapsed();
    if (this->verbose()) Feel::FeelModels::Log(this->prefix()+".FluidMechanics","updateJacobian",
                                               "assemble convection term in "+(boost::format("%1% s") % timeElapsed).str(),
                                               this->worldComm(),this->verboseAllProc());

    //--------------------------------------------------------------------------------------------------//
    // sigma : grad(v) on Omega
    if ( this->densityViscosityModel()->dynamicViscosityLaw() == "newtonian")
    {
        //auto const deft = sym(gradt(u));
        //--------------------------------------------------------------------------------------------------//
        // newtonian law
        auto const& mu = this->densityViscosityModel()->fieldMu();
        auto const sigma_newtonian_viscous = idv(mu)*deft;
        auto const Sigmat_newtonian = -idt(p)*Id + 2*idv(mu)*deft;
        //--------------------------------------------------------------------------------------------------//
        if ( BuildCstPart )
        {
#if 1
            bilinearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= inner(Sigmat_newtonian,grad(v)),
                           _geomap=this->geomap() );
#else
            //auto StressTensorExprJac = Feel::vf::FSI::fluidMecNewtonianStressTensorJacobian(u,p,viscosityModel,false/*true*/);
            bilinearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= 2*idv(mu)*inner(deft,grad(v)),
                           //_expr= inner( StressTensorExprJac, grad(v) ),
                           _geomap=this->geomap() );
            bilinearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= -idt(p)*div(v),
                           _geomap=this->geomap() );
#endif

        }
    }
    else
    {
        if ( BuildCstPart )
            bilinearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= -idt(p)*div(v),
                           _geomap=this->geomap() );

        if ( BuildNonCstPart )
        {
            auto StressTensorExprJac = Feel::vf::FeelModels::fluidMecNewtonianStressTensorJacobian<2*nOrderVelocity>(u,p,*this->densityViscosityModel(),false/*true*/);
            bilinearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           //_expr= inner( 2*sigma_powerlaw_viscous/*Sigmat_powerlaw*/,grad(v) ),
                           _expr= inner( StressTensorExprJac,grad(v) ),
                           _geomap=this->geomap() );
        }
    } // non newtonian


    //--------------------------------------------------------------------------------------------------//
    // incompressibility term
    if (BuildCstPart)
    {
        bilinearForm_PatternCoupled +=
            integrate( _range=M_rangeMeshElements,
                       _expr= -idv(rho)*divt(u)*id(q),
                       _geomap=this->geomap() );
    }

    //--------------------------------------------------------------------------------------------------//
    //transients terms
    bool Build_TransientTerm = !BuildCstPart;
    if ( this->timeStepBase()->strategy()==TS_STRATEGY_DT_CONSTANT ) Build_TransientTerm=BuildCstPart;

    if (!this->isStationaryModel() && Build_TransientTerm/*BuildCstPart*/)
    {
        bilinearForm_PatternDefault +=
            integrate( _range=M_rangeMeshElements,
                       _expr= idv(rho)*trans(idt(u))*id(v)*M_bdf_fluid->polyDerivCoefficient(0),
                       _geomap=this->geomap() );
    }

    //--------------------------------------------------------------------------------------------------//
    // user-defined additional terms
    this->updateJacobianAdditional( J, _BuildCstPart );

    //--------------------------------------------------------------------------------------------------//
    // define pressure cst
    if ( this->definePressureCst() )
    {
        if ( this->definePressureCstMethod() == "penalisation" && BuildCstPart )
        {
            double beta = this->definePressureCstPenalisationBeta();
            for ( auto const& rangeElt : M_definePressureCstMeshRanges )
                bilinearForm_PatternCoupled +=
                    integrate( _range=rangeElt,
                               _expr=beta*idt(p)*id(q),
                               _geomap=this->geomap() );
        }
        if ( this->definePressureCstMethod() == "lagrange-multiplier" && BuildCstPart )
        {
            CHECK( this->startBlockIndexFieldsInMatrix().find("define-pressure-cst-lm") != this->startBlockIndexFieldsInMatrix().end() )
                << " start dof index for define-pressure-cst-lm is not present\n";
            size_type startBlockIndexDefinePressureCstLM = this->startBlockIndexFieldsInMatrix().find("define-pressure-cst-lm")->second;
            for ( int k=0;k<M_XhMeanPressureLM.size();++k )
            {
                auto lambda = M_XhMeanPressureLM[k]->element();
                form2( _test=Xh, _trial=M_XhMeanPressureLM[k], _matrix=J,
                       _rowstart=this->rowStartInMatrix(),
                       _colstart=this->colStartInMatrix()+startBlockIndexDefinePressureCstLM+k ) +=
                    integrate( _range=M_definePressureCstMeshRanges[k],
                               _expr= id(p)*idt(lambda) /*+ idt(p)*id(lambda)*/,
                               _geomap=this->geomap() );

                form2( _test=M_XhMeanPressureLM[k], _trial=Xh, _matrix=J,
                       _rowstart=this->rowStartInMatrix()+startBlockIndexDefinePressureCstLM+k,
                       _colstart=this->colStartInMatrix() ) +=
                    integrate( _range=M_definePressureCstMeshRanges[k],
                               _expr= + idt(p)*id(lambda),
                               _geomap=this->geomap() );
            }
        }
    }

    //--------------------------------------------------------------------------------------------------//

    this->updateJacobianStabilisation( data, U );

    //--------------------------------------------------------------------------------------------------//

    this->updateJacobianWeakBC( data, U );

    //--------------------------------------------------------------------------------------------------//
    if ( M_useThermodynModel && M_useGravityForce )
    {
        DataUpdateJacobian dataThermo( data );
        dataThermo.setDoBCStrongDirichlet( false );
        M_thermodynModel->updateJacobian( dataThermo );

        if ( BuildNonCstPart )
        {
            auto XhT = M_thermodynModel->spaceTemperature();
            auto t = XhT->element(XVec, M_thermodynModel->rowStartInVector() );
            auto const& thermalProperties = M_thermodynModel->thermalProperties();

            auto thecoeff = idv(thermalProperties->fieldRho())*idv(thermalProperties->fieldHeatCapacity());
            form2( _test=XhT,_trial=XhT,_matrix=J,
                   _rowstart=M_thermodynModel->rowStartInMatrix(),
                   _colstart=M_thermodynModel->colStartInMatrix() ) +=
                integrate( _range=M_rangeMeshElementsAeroThermal,
                           _expr= thecoeff*(gradt(t)*idv(u))*id(t),
                       _geomap=this->geomap() );
            form2( _test=XhT,_trial=Xh,_matrix=J,
                   _rowstart=M_thermodynModel->rowStartInMatrix(),
                   _colstart=this->colStartInMatrix() ) +=
                integrate( _range=M_rangeMeshElementsAeroThermal,
                           _expr= thecoeff*(gradv(t)*idt(u))*id(t),
                           _geomap=this->geomap() );

            auto betaFluid = idv(thermalProperties->fieldThermalExpansion() );
            form2( _test=Xh,_trial=XhT,_matrix=J,
                   _rowstart=this->rowStartInMatrix(),
                   _colstart=M_thermodynModel->colStartInMatrix() ) +=
                integrate( _range=M_rangeMeshElementsAeroThermal,
                           _expr= idv(thermalProperties->fieldRho())*betaFluid*(idt(t))*inner(M_gravityForce,id(u)),
                           _geomap=this->geomap() );
        }


    }
    //--------------------------------------------------------------------------------------------------//

    if ( BuildNonCstPart && _doBCStrongDirichlet )
    {
        if ( this->hasMarkerDirichletBCelimination() )
            this->updateJacobianStrongDirichletBC(J,RBis);

        std::list<std::string> markerDirichletEliminationOthers;
#if defined( FEELPP_MODELS_HAS_MESHALE )
        if (this->isMoveDomain() && this->couplingFSIcondition()=="dirichlet-neumann")
        {
            for (std::string const& marker : this->markersNameMovingBoundary() )
                markerDirichletEliminationOthers.push_back( marker );
        }
#endif
        for ( auto const& inletbc : M_fluidInletDesc )
        {
            std::string const& marker = std::get<0>( inletbc );
            markerDirichletEliminationOthers.push_back( marker );
        }

        if ( !markerDirichletEliminationOthers.empty() )
            bilinearForm_PatternCoupled +=
                on( _range=markedfaces(mesh, markerDirichletEliminationOthers ),
                    _element=u,_rhs=RBis,
                    _expr= vf::zero<nDim,1>() );


        if ( this->hasMarkerPressureBC() )
        {
            size_type startBlockIndexPressureLM1 = this->startBlockIndexFieldsInMatrix().find("pressurelm1")->second;
            form2( _test=M_spaceLagrangeMultiplierPressureBC,_trial=M_spaceLagrangeMultiplierPressureBC,_matrix=J,
                   _rowstart=rowStartInMatrix+startBlockIndexPressureLM1,
                   _colstart=rowStartInMatrix+startBlockIndexPressureLM1 ) +=
                on( _range=boundaryfaces(M_meshLagrangeMultiplierPressureBC), _rhs=RBis,
                    _element=*M_fieldLagrangeMultiplierPressureBC1, _expr=cst(0.));
            if ( nDim == 3 )
            {
                size_type startBlockIndexPressureLM2 = this->startBlockIndexFieldsInMatrix().find("pressurelm2")->second;
                form2( _test=M_spaceLagrangeMultiplierPressureBC,_trial=M_spaceLagrangeMultiplierPressureBC,_matrix=J,
                       _rowstart=rowStartInMatrix+startBlockIndexPressureLM2,
                       _colstart=rowStartInMatrix+startBlockIndexPressureLM2 ) +=
                    on( _range=boundaryfaces(M_meshLagrangeMultiplierPressureBC), _rhs=RBis,
                        _element=*M_fieldLagrangeMultiplierPressureBC2, _expr=cst(0.));
            }
        }


        if ( M_useThermodynModel && M_useGravityForce )
        {
            M_thermodynModel->updateBCStrongDirichletJacobian( J,RBis );
        }

    }

    //--------------------------------------------------------------------------------------------------//

    /*double*/ timeElapsed=thetimer.elapsed();
    if (this->verbose()) Feel::FeelModels::Log(this->prefix()+".FluidMechanics","updateJacobian",
                                               "finish"+sc+" in "+(boost::format("%1% s") % timeElapsed).str()+
                                               "\n--------------------------------------------------",
                                               this->worldComm(),this->verboseAllProc());

} // updateJacobian

} // namespace FeelModels

} // namespace Feel


