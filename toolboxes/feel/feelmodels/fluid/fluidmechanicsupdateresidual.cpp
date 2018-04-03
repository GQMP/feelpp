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
FLUIDMECHANICS_CLASS_TEMPLATE_TYPE::updateResidual( DataUpdateResidual & data ) const
{
    using namespace Feel::vf;
    const vector_ptrtype& XVec = data.currentSolution();
    vector_ptrtype& R = data.residual();
    bool BuildCstPart = data.buildCstPart();
    bool UseJacobianLinearTerms = data.useJacobianLinearTerms();
    bool _doBCStrongDirichlet = data.doBCStrongDirichlet();
    bool BuildNonCstPart = !BuildCstPart;

    std::string sc=(BuildCstPart)?" (build cst part)":" (build non cst part)";
    if (this->verbose()) Feel::FeelModels::Log("--------------------------------------------------\n",
                                               this->prefix()+".FluidMechanics","updateResidual","start"+sc,
                                               this->worldComm(),this->verboseAllProc());

    boost::mpi::timer thetimer, thetimerBis;


    //--------------------------------------------------------------------------------------------------//

    auto mesh = this->mesh();
    auto Xh = this->functionSpace();

    size_type rowStartInVector = this->rowStartInVector();
    auto linearForm_PatternDefault = form1( _test=Xh, _vector=R,
                                            _pattern=size_type(Pattern::DEFAULT),
                                            _rowstart=rowStartInVector );
    auto linearForm_PatternCoupled = form1( _test=Xh, _vector=R,
                                            _pattern=size_type(Pattern::COUPLED),
                                            _rowstart=rowStartInVector );

    auto U = Xh->element(XVec, rowStartInVector);
    auto u = U.template element<0>();
    auto v = U.template element<0>();
    auto p = U.template element<1>();
    auto q = U.template element<1>();

    //--------------------------------------------------------------------------------------------------//

    // strain tensor (trial)
    auto defv = sym(gradv(u));
    // identity Matrix
    auto Id = eye<nDim,nDim>();
    // dynamic viscosity
    auto const& mu = this->densityViscosityModel()->fieldMu();
    auto const& rho = this->densityViscosityModel()->fieldRho();
    // stress tensor (eval)
    auto Sigmav = -idv(p)*Id + 2*idv(mu)*defv;

    double timeElapsedBis=thetimerBis.elapsed();
    this->log("FluidMechanics","updateResidual","init done in "+(boost::format("%1% s") % timeElapsedBis ).str() );

    //--------------------------------------------------------------------------------------------------//

    thetimerBis.restart();
    if ( BuildNonCstPart )
    {
        if ( this->doStabConvectionEnergy() )
        {
            linearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           //_expr= /*idv(*M_P0Rho)**/inner( Feel::vf::FSI::fluidMecConvection(u,*M_P0Rho) + idv(*M_P0Rho)*0.5*divv(u)*idv(u), id(v) ),
                           _expr=inner( Feel::vf::FeelModels::fluidMecConvectionWithEnergyStab(u,rho), id(v) ),
                           _geomap=this->geomap() );

            /*if (this->isMoveDomain()  && !BuildCstPart && !UseJacobianLinearTerms)
             {
             linearForm_PatternCoupled +=
             integrate( _range=elements(mesh),
             _expr= -0.5*idv(M_P0Rho)*divv(this->meshVelocity())*trans(idv(u))*id(v),
             _geomap=this->geomap() );
             }*/
        }
        else
        {
            // convection term
#if 0
            auto convecTerm = val( idv(rho)*trans( gradv(u)*idv(u) ))*id(v);
#else
            auto convecTerm = inner( Feel::vf::FeelModels::fluidMecConvection(u,rho),id(v) );
#endif
            linearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr=convecTerm,
                           _geomap=this->geomap() );
        }

    }

    timeElapsedBis=thetimerBis.elapsed();thetimerBis.restart();
    this->log("FluidMechanics","updateResidual","build convective--1-- term in "+(boost::format("%1% s") % timeElapsedBis ).str() );

#if defined( FEELPP_MODELS_HAS_MESHALE )
    if ( M_isMoveDomain && !BuildCstPart && !UseJacobianLinearTerms )
    {
        // mesh velocity (convection) term
        linearForm_PatternCoupled +=
            integrate( _range=M_rangeMeshElements,
                       _expr= -val(idv(rho)*trans( gradv(u)*( idv( this->meshVelocity() ))))*id(v),
                       _geomap=this->geomap() );
        timeElapsedBis=thetimerBis.elapsed();
        this->log("FluidMechanics","updateResidual","build convective--2-- term in "+(boost::format("%1% s") % timeElapsedBis ).str() );
    }
#endif

    //--------------------------------------------------------------------------------------------------//
    //this->updateResidualModel( data, U );
    if ( this->densityViscosityModel()->dynamicViscosityLaw() == "newtonian")
    {
        // sigma : grad(v) on Omega
        if ( BuildNonCstPart && !UseJacobianLinearTerms )
        {
            this->log("FluidMechanics","updateResidualModel","assembly with newtonian viscosity" );
            auto const mu_newtonian = idv(mu);
            auto const Sigmav_newtonian = -idv(p)*Id + 2*mu_newtonian*defv;
#if 1
            linearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           //_expr= inner( StressTensorExpr,grad(v) ),
                           _expr= inner( val(Sigmav_newtonian),grad(v) ),
                           _geomap=this->geomap() );
#else
            form1( Xh, R ) +=
                integrate( _range=M_rangeMeshElements,
                           _expr= 2*idv(*M_P0Mu)*trace(trans(defv)*grad(v)),
                           _geomap=this->geomap() );
            form1( Xh, R ) +=
                integrate( _range=M_rangeMeshElements,
                           _expr= -idv(p)*div(v),
                           _geomap=this->geomap() );
#endif
        }
    }
    else
    {
        if ( BuildNonCstPart && !UseJacobianLinearTerms )
        {
            linearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= -idv(p)*div(v),
                           _geomap=this->geomap() );
        }
        if ( BuildNonCstPart )
        {
            auto const StressTensorExpr = Feel::vf::FeelModels::fluidMecNewtonianStressTensor<2*nOrderVelocity>(u,p,*this->densityViscosityModel(),false/*true*/);
            // sigma : grad(v) on Omega
            linearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= inner( StressTensorExpr,grad(v) ),
                           _geomap=this->geomap() );
        }
    } // non newtonian

    //--------------------------------------------------------------------------------------------------//
    // take into account that div u != 0
    if (!this->velocityDivIsEqualToZero() && BuildCstPart)
    {
        linearForm_PatternCoupled +=
            integrate( _range=M_rangeMeshElements,
                       _expr= idv(this->velocityDiv())*id(q),
                       _geomap=this->geomap() );

        auto coeffDiv = (2./3.)*idv(this->densityViscosityModel()->fieldMu()); //(eps-2mu/3)
        linearForm_PatternCoupled +=
            integrate( _range=M_rangeMeshElements,
                       _expr= val(-coeffDiv*gradv(this->velocityDiv()))*id(v),
                       _geomap=this->geomap() );
    }

    //--------------------------------------------------------------------------------------------------//

    // incompressibility term
    if (!BuildCstPart && !UseJacobianLinearTerms )
    {
        linearForm_PatternCoupled +=
            integrate( _range=M_rangeMeshElements,
                       _expr= -idv(rho)*divv(u)*id(q),
                       _geomap=this->geomap() );
    }

    //--------------------------------------------------------------------------------------------------//

    // body forces
    if (BuildCstPart)
    {
        if ( this->M_overwritemethod_updateSourceTermResidual != NULL )
        {
            this->M_overwritemethod_updateSourceTermResidual(R);
        }
        else
        {
            for( auto const& d : this->M_volumicForcesProperties )
            {
                auto rangeBodyForceUsed = ( marker(d).empty() )? M_rangeMeshElements : markedelements(this->mesh(),marker(d));
                linearForm_PatternCoupled +=
                    integrate( _range=rangeBodyForceUsed,
                               _expr= -inner( expression(d),id(v) ),
                               _geomap=this->geomap() );
            }
        }

        if (M_haveSourceAdded)
        {
            linearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= -trans(idv(*M_SourceAdded))*id(v),
                           _geomap=this->geomap() );
        }
        if ( M_useGravityForce )
        {
            linearForm_PatternCoupled +=
                integrate( _range=M_rangeMeshElements,
                           _expr= -idv(rho)*inner(M_gravityForce,id(u)),
                           _geomap=this->geomap() );
        }
    }

    //------------------------------------------------------------------------------------//

    //transients terms
    if (!this->isStationaryModel())
    {
        bool Build_TransientTerm = !BuildCstPart;
        if ( this->timeStepBase()->strategy()==TS_STRATEGY_DT_CONSTANT ) Build_TransientTerm=!BuildCstPart && !UseJacobianLinearTerms;

        if (Build_TransientTerm) //  !BuildCstPart && !UseJacobianLinearTerms )
        {
            linearForm_PatternDefault +=
                integrate( _range=M_rangeMeshElements,
                           _expr= val(idv(rho)*trans(idv(u))*M_bdf_fluid->polyDerivCoefficient(0))*id(v),
                           _geomap=this->geomap() );
        }

        if (BuildCstPart)
        {
            auto Buzz = M_bdf_fluid->polyDeriv();
            auto buzz = Buzz.template element<0>();
            linearForm_PatternDefault +=
                integrate( _range=M_rangeMeshElements,
                           _expr= val(-idv(rho)*trans(idv(buzz)))*id(v),
                           _geomap=this->geomap() );
        }
    }

    //--------------------------------------------------------------------------------------------------//
    // user-defined additional terms
    this->updateResidualAdditional( R, BuildCstPart );

    //------------------------------------------------------------------------------------//
    // define pressure cst
    if ( this->definePressureCst() )
    {
        if ( this->definePressureCstMethod() == "penalisation" && !BuildCstPart && !UseJacobianLinearTerms )
        {
            double beta = this->definePressureCstPenalisationBeta();
            for ( auto const& rangeElt : M_definePressureCstMeshRanges )
                linearForm_PatternCoupled +=
                    integrate( _range=rangeElt,
                               _expr=beta*idv(p)*id(q),
                               _geomap=this->geomap() );
        }
        if ( this->definePressureCstMethod() == "lagrange-multiplier" )
        {
            CHECK( this->startBlockIndexFieldsInMatrix().find("define-pressure-cst-lm") != this->startBlockIndexFieldsInMatrix().end() )
                << " start dof index for define-pressure-cst-lm is not present\n";
            size_type startBlockIndexDefinePressureCstLM = this->startBlockIndexFieldsInMatrix().find("define-pressure-cst-lm")->second;

            if ( !BuildCstPart && !UseJacobianLinearTerms )
            {
                for ( int k=0;k<M_XhMeanPressureLM.size();++k )
                {
                    auto lambda = M_XhMeanPressureLM[k]->element(XVec,rowStartInVector+startBlockIndexDefinePressureCstLM+k);
                    //M_blockVectorSolution.setSubVector( lambda, *XVec, rowStartInVector+startBlockIndexDefinePressureCstLM+k );
                    //for ( size_type k=0;k<M_XhMeanPressureLM->nLocalDofWithGhost();++k )
                    //    lambda( k ) = XVec->operator()( startDofIndexDefinePressureCstLM + k);

                    form1( _test=M_XhMeanPressureLM[k],_vector=R,
                           _rowstart=rowStartInVector+startBlockIndexDefinePressureCstLM+k ) +=
                        integrate( _range=M_definePressureCstMeshRanges[k],
                                   _expr= id(p)*idv(lambda) + idv(p)*id(lambda),
                                   _geomap=this->geomap() );
                }
            }
#if defined(FLUIDMECHANICS_USE_LAGRANGEMULTIPLIER_MEANPRESSURE)
            if ( BuildCstPart )
            {
                for ( int k=0;k<M_XhMeanPressureLM.size();++k )
                {
                    auto lambda = M_XhMeanPressureLM[k]->element();
                    form1( _test=M_XhMeanPressureLM[k],_vector=R,
                           _rowstart=rowStartInVector+startDofIndexDefinePressureCstLM+k ) +=
                        integrate( _range=M_definePressureCstMeshRanges[k],
                                   _expr= -(FLUIDMECHANICS_USE_LAGRANGEMULTIPLIER_MEANPRESSURE(this->shared_from_this()))*id(lambda),
                                   _geomap=this->geomap() );
                }
            }
#endif
        }
    }


    //------------------------------------------------------------------------------------//

    this->updateResidualStabilisation( data, U );

    //------------------------------------------------------------------------------------//
#if 0
    if ( UsePeriodicity && !BuildCstPart )
    {
        std::string marker1 = soption(_name="periodicity.marker1",_prefix=this->prefix());
        double pressureJump = doption(_name="periodicity.pressure-jump",_prefix=this->prefix());
        linearForm_PatternCoupled +=
            integrate( _range=markedfaces( this->mesh(),this->mesh()->markerName(marker1) ),
                       _expr=-inner(pressureJump*N(),id(v) ) );
    }
#endif
    //------------------------------------------------------------------------------------//

    this->updateResidualWeakBC( data, U );

    //------------------------------------------------------------------------------------//

    if (!BuildCstPart && _doBCStrongDirichlet && this->hasStrongDirichletBC() )
    {
        R->close();

        this->updateResidualStrongDirichletBC( R );
    }

    //------------------------------------------------------------------------------------//

    double timeElapsed=thetimer.elapsed();
    if (this->verbose()) Feel::FeelModels::Log(this->prefix()+".FluidMechanics","updateResidual",
                                               "finish"+sc+" in "+(boost::format("%1% s") % timeElapsed).str()+
                                               "\n--------------------------------------------------",
                                               this->worldComm(),this->verboseAllProc());


} // updateResidual

FLUIDMECHANICS_CLASS_TEMPLATE_DECLARATIONS
void
FLUIDMECHANICS_CLASS_TEMPLATE_TYPE::updateNewtonInitialGuess(vector_ptrtype& U) const
{
    this->log("FluidMechanics","updateNewtonInitialGuess","start");

    size_type rowStartInVector = this->rowStartInVector();
    auto Xh = this->functionSpace();
    auto up = Xh->element( U, rowStartInVector );
    auto u = up.template element<0>();
    auto mesh = this->mesh();

#if defined( FEELPP_MODELS_HAS_MESHALE )
    if (this->isMoveDomain() && this->couplingFSIcondition()=="dirichlet-neumann")
    {
        this->log("FluidMechanics","updateNewtonInitialGuess","update moving boundary with strong Dirichlet");
        u.on(_range=markedfaces(mesh, this->markersNameMovingBoundary()),
             _expr=idv(this->meshVelocity2()) );
    }
#endif

    if ( this->hasMarkerDirichletBCelimination() )
    {
        // store markers for each entities in order to apply strong bc with priority (points erase edges erace faces)
        std::map<std::string, std::tuple< std::list<std::string>,std::list<std::string>,std::list<std::string>,std::list<std::string> > > mapMarkerBCToEntitiesMeshMarker;
        for( auto const& d : M_bcDirichlet )
        {
            mapMarkerBCToEntitiesMeshMarker[marker(d)] =
                detail::distributeMarkerListOnSubEntity( mesh, this->markerDirichletBCByNameId( "elimination",marker(d) ) );
        }
        std::map<std::pair<std::string,ComponentType>, std::tuple< std::list<std::string>,std::list<std::string>,std::list<std::string>,std::list<std::string> > > mapCompMarkerBCToEntitiesMeshMarker;
        for ( auto const& bcDirComp : M_bcDirichletComponents )
        {
            ComponentType comp = bcDirComp.first;
            for( auto const& d : bcDirComp.second )
            {
                mapCompMarkerBCToEntitiesMeshMarker[std::make_pair(marker(d),comp)] =
                    detail::distributeMarkerListOnSubEntity(mesh, this->markerDirichletBCByNameId( "elimination",marker(d), comp )   );
            }
        }

        // strong Dirichlet bc with vectorial velocity
        for( auto const& d : M_bcDirichlet )
        {
            auto itFindMarker = mapMarkerBCToEntitiesMeshMarker.find( marker(d) );
            if ( itFindMarker == mapMarkerBCToEntitiesMeshMarker.end() )
                continue;
            auto const& listMarkerFaces = std::get<0>( itFindMarker->second );
            if ( !listMarkerFaces.empty() )
                u.on(_range=markedfaces(mesh,listMarkerFaces ),
                     _expr=expression(d) );
            auto const& listMarkerEdges = std::get<1>( itFindMarker->second );
            if ( !listMarkerEdges.empty() )
                u.on(_range=markededges(mesh,listMarkerEdges ),
                     _expr=expression(d) );
            auto const& listMarkerPoints = std::get<2>( itFindMarker->second );
            if ( !listMarkerPoints.empty() )
                u.on(_range=markedpoints(mesh,listMarkerPoints ),
                     _expr=expression(d) );
        }
        // strong Dirichlet bc with velocity components
        for ( auto const& bcDirComp : M_bcDirichletComponents )
        {
            ComponentType comp = bcDirComp.first;
            for( auto const& d : bcDirComp.second )
            {
                auto itFindMarker = mapCompMarkerBCToEntitiesMeshMarker.find( std::make_pair(marker(d),comp) );
                if ( itFindMarker == mapCompMarkerBCToEntitiesMeshMarker.end() )
                    continue;
                auto const& listMarkerFaces = std::get<0>( itFindMarker->second );
                if ( !listMarkerFaces.empty() )
                    u[comp].on(_range=markedfaces(mesh,listMarkerFaces ),
                               _expr=expression(d) );
                auto const& listMarkerEdges = std::get<1>( itFindMarker->second );
                if ( !listMarkerEdges.empty() )
                    u[comp].on(_range=markededges(mesh,listMarkerEdges ),
                               _expr=expression(d) );
                auto const& listMarkerPoints = std::get<2>( itFindMarker->second );
                if ( !listMarkerPoints.empty() )
                    u[comp].on(_range=markedpoints(mesh,listMarkerPoints ),
                               _expr=expression(d) );
            }
        }
    }

    for ( auto const& inletbc : M_fluidInletDesc )
    {
        std::string const& marker = std::get<0>( inletbc );
        auto itFindMark = M_fluidInletVelocityInterpolated.find(marker);
        if ( itFindMark == M_fluidInletVelocityInterpolated.end() )
            continue;
        auto const& inletVel = std::get<0>( itFindMark->second );
        u.on(_range=markedfaces(mesh, marker),
             _expr=-idv(inletVel)*N() );
    }
    // synchronize velocity dof on interprocess
    auto itFindDofsWithValueImposed = M_dofsWithValueImposed.find("velocity");
    if ( itFindDofsWithValueImposed != M_dofsWithValueImposed.end() )
        sync( u, "=", itFindDofsWithValueImposed->second );

    if ( this->definePressureCst() && this->definePressureCstMethod() == "algebraic" )
    {
        auto upSol = this->functionSpace()->element( U, this->rowStartInVector() );
        auto pSol = upSol.template element<1>();
        CHECK( !M_definePressureCstAlgebraicOperatorMeanPressure.empty() ) << "mean pressure operator does not init";

        for ( int k=0;k<M_definePressureCstAlgebraicOperatorMeanPressure.size();++k )
        {
            double meanPressureImposed = 0;
            double meanPressureCurrent = inner_product( *M_definePressureCstAlgebraicOperatorMeanPressure[k].first, pSol );
            for ( size_type dofId : M_definePressureCstAlgebraicOperatorMeanPressure[k].second )
                pSol(dofId) += (meanPressureImposed - meanPressureCurrent);
        }
        sync( pSol, "=" );
    }

    this->log("FluidMechanics","updateNewtonInitialGuess","finish");
}

FLUIDMECHANICS_CLASS_TEMPLATE_DECLARATIONS
void
FLUIDMECHANICS_CLASS_TEMPLATE_TYPE::updateResidualStrongDirichletBC( vector_ptrtype& R ) const
{
    auto Xh = this->spaceVelocityPressure();
    size_type rowStartInVector = this->rowStartInVector();

    auto resFeView = Xh->element(R,rowStartInVector);
    auto resFeViewVelocity = resFeView.template element<0>();

    auto itFindDofsWithValueImposed = M_dofsWithValueImposed.find("velocity");
    auto const& dofsWithValueImposedVelocity = ( itFindDofsWithValueImposed != M_dofsWithValueImposed.end() )? itFindDofsWithValueImposed->second : std::set<size_type>();
    for ( size_type thedof : dofsWithValueImposedVelocity )
        resFeViewVelocity.set( thedof,0. );
    sync( resFeViewVelocity, "=", dofsWithValueImposedVelocity );


    if ( this->hasMarkerPressureBC() )
    {
#if 0
        size_type startBlockIndexPressureLM1 = this->startBlockIndexFieldsInMatrix().find("pressurelm1")->second;
        auto lambdaPressure1 = M_spaceLagrangeMultiplierPressureBC->element( R/*XVec*/, rowStartInVector+startBlockIndexPressureLM1 );
        lambdaPressure1.on(_range=boundaryfaces(M_meshLagrangeMultiplierPressureBC),
                           _expr=vf::zero<1,1>() );
        if ( nDim == 3 )
        {
            size_type startBlockIndexPressureLM2 = this->startBlockIndexFieldsInMatrix().find("pressurelm2")->second;
            auto lambdaPressure2 = M_spaceLagrangeMultiplierPressureBC->element( R/*XVec*/, rowStartInVector+startBlockIndexPressureLM2 );
            lambdaPressure2.on(_range=boundaryfaces(M_meshLagrangeMultiplierPressureBC),
                               _expr=vf::zero<1,1>() );
        }
#else
        auto itFindDofsWithValueImposedPressureBC = M_dofsWithValueImposed.find("pressurebc-lm");
        auto const& dofsWithValueImposedPressureBC = ( itFindDofsWithValueImposedPressureBC != M_dofsWithValueImposed.end() )? itFindDofsWithValueImposedPressureBC->second : std::set<size_type>();
        size_type startBlockIndexPressureLM1 = this->startBlockIndexFieldsInMatrix().find("pressurelm1")->second;
        auto lambdaPressure1 = M_spaceLagrangeMultiplierPressureBC->element( R/*XVec*/, rowStartInVector+startBlockIndexPressureLM1 );
        for ( size_type thedof : dofsWithValueImposedPressureBC )
            lambdaPressure1.set( thedof,0. );
        sync( lambdaPressure1, "=", dofsWithValueImposedPressureBC );
        if ( nDim == 3 )
        {
            size_type startBlockIndexPressureLM2 = this->startBlockIndexFieldsInMatrix().find("pressurelm2")->second;
            auto lambdaPressure2 = M_spaceLagrangeMultiplierPressureBC->element( R/*XVec*/, rowStartInVector+startBlockIndexPressureLM2 );
            for ( size_type thedof : dofsWithValueImposedPressureBC )
                lambdaPressure2.set( thedof,0. );
            sync( lambdaPressure2, "=", dofsWithValueImposedPressureBC );
        }

#endif
    }

}




} // namespace FeelModels
} // namespace Feel


