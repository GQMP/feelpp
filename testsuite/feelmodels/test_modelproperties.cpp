/* -*- mode: c++; coding: utf-8; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; show-trailing-whitespace: t  -*-

 This file is part of the Feel++ library

 Author(s): Christophe Prud'homme <christophe.prudhomme@feelpp.org>
 Date: 15 Mar 2015

 Copyright (C) 2015 Feel++ Consortium

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */
#define BOOST_TEST_MODULE model properties testsuite

#include <testsuite.hpp>

#include <feel/feelcore/environment.hpp>
#include <feel/feelmodels/modelproperties.hpp>
#include <feel/feeldiscr/pch.hpp>
#include <feel/feelfilters/loadmesh.hpp>
#include <feel/feelvf/vf.hpp>

using namespace Feel;

inline
po::options_description
makeOptions()
{
    po::options_description modelopt( "test model properties options" );
    modelopt.add_options()
        ("json_filename" , Feel::po::value<std::string>()->default_value( "$cfgdir/test.feelpp" ),
         "json file" )
        ;
    return  modelopt.add( Feel::feel_options() ) ;
}

inline
AboutData
makeAbout()
{
    AboutData about( "test_modelproperties",
                     "test_modelproperties" );
    return about;

}

FEELPP_ENVIRONMENT_WITH_OPTIONS( makeAbout(), makeOptions() )

BOOST_AUTO_TEST_SUITE( modelproperties )

BOOST_AUTO_TEST_CASE( test_materials )
{
    auto mesh = loadMesh(new Mesh<Simplex<3> >);
    auto Xh = Pch<1>(mesh);
    auto g = Xh->element();
    auto d = Xh->element();

    ModelProperties model_props( Environment::expand(soption("json_filename")) );
    auto mats = model_props.materials();
    for ( auto matPair : mats )
    {
        auto mat = matPair.second;
        auto physics = mat.physics();
        auto name = mat.getString("name");
        auto rhoInt = mat.getInt("rho");
        auto etaDouble = mat.getDouble("eta");

        auto rho = mat.getScalar( "rho" );
        auto f = mat.getScalar("f","g",idv(g));
        auto fParam = mat.getScalar("f",{{"g",1.}});
        auto hList = mat.getScalar("h",{"g","t"},{idv(g),idv(d)});
        auto fVec = mat.getScalar("f",std::vector<std::string>(1,"g"), std::vector<decltype(idv(g))>(1,idv(g)));
        auto hParams = mat.getScalar("h",{"g"},{idv(g)}, {{"t",1.}});

        auto nu = mat.getVector<3>( "nu" );
        auto curlnu = curl(nu);
        auto muPair = mat.getVector<2>("mu",{"t",1.});
        auto muMap = mat.getVector<2>("mu", {{"t",1.}});

        auto chi = mat.getMatrix<2>( "chi" );
        auto xhiPair = mat.getMatrix<3>( "xhi", {"t",2.} );
        auto xhiMap = mat.getMatrix<3,3>( "xhi", {{"t",3.}});

#if 0
        Feel::cout << "properties for " << matPair.first << std::endl;
        Feel::cout << "\t" << name << std::endl;
        Feel::cout << "\thas " << physics.size() << " physics:" << std::endl;
        for( auto const& p : physics )
            Feel::cout << "\t\t" << p << std::endl;
        Feel::cout << "\t" << rhoInt << std::endl;
        Feel::cout << "\t" << etaDouble << std::endl;
        Feel::cout << "\t" << rho << std::endl;
        Feel::cout << "\t" << nu << std::endl;
        Feel::cout << "\t" << curlnu << std::endl;
        Feel::cout << "\t" << muPair << std::endl;
        Feel::cout << "\t" << muMap << std::endl;
        Feel::cout << "\t" << chi << std::endl;
        Feel::cout << "\t" << xhiPair << std::endl;
        Feel::cout << "\t" << xhiMap << std::endl;
#endif
    }
#if 0
    Feel::cout << mats.materialWithPhysic("electro").size() << " materials with electro physic" << std::endl;
    Feel::cout << mats.materialWithPhysic("thermo").size() << " materials with thermo physic" << std::endl;
#endif
    BOOST_CHECK_EQUAL(mats.materialWithPhysic("electro").size(), 2);
    BOOST_CHECK_EQUAL(mats.materialWithPhysic("thermo").size(), 1);

}

BOOST_AUTO_TEST_CASE( test_parameters )
{
    ModelProperties model_props( Environment::expand(soption("json_filename")) );
    auto param = model_props.parameters();
    for ( auto const& pp : param )
    {
        auto p = pp.second;
        if( p.name() == "Um" )
        {
            BOOST_CHECK_CLOSE( p.value(), 0.3, 10e-8);
            BOOST_CHECK_CLOSE( p.min(), 1e-4, 10e-8);
            BOOST_CHECK_EQUAL( p.max(), 10);
        }
        else if( p.name() == "H" )
        {
            BOOST_CHECK_CLOSE( p.value(), 0.41, 10e-8 );
        }
#if 0
        Feel::cout << p.name() << std::endl
                   << "\tvalue : " << p.value() << std::endl
                   << "\tmin   : " << p.min() << std::endl
                   << "\tmax   : " << p.max() << std::endl;
        if ( p.hasExpression() )
            Feel::cout << "\texpr  : " << p.expression() << std::endl;
#endif
    }
}

BOOST_AUTO_TEST_CASE( test_outputs )
{
    ModelProperties model_props( Environment::expand(soption("json_filename")) );
    auto outputs = model_props.outputs();
    for( auto const& out : outputs )
    {
        auto output = out.second;
        if( output.name() == "myoutput" )
        {
            BOOST_CHECK_EQUAL( output.type(), "average");
            BOOST_CHECK_EQUAL( output.range().size(), 2);
            BOOST_CHECK_EQUAL( output.dim(), 3 );
        }
        else if( output.type() == "flux" )
        {
            BOOST_CHECK_EQUAL( output.type(), "flux");
            BOOST_CHECK_EQUAL( output.range().size(), 1);
            BOOST_CHECK_EQUAL( output.dim(), 2 );
        }
#if 0
        std::cout << output;
#endif
    }
}

BOOST_AUTO_TEST_SUITE_END()
