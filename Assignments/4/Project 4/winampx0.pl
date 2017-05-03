#!/usr/bin/perl

binmode STDOUT;

$| = 1;

$length = "\xFF\xFF";
$function_name = "Aa0Aa1Aa2Aa3Aa4Aa5Aa6Aa7Aa8Aa9Ab0Ab1Ab2Ab3Ab4Ab5Ab6Ab7Ab8Ab9Ac0Ac1Ac2Ac3Ac4Ac5Ac6Ac7Ac8Ac9Ad0Ad1Ad2Ad3Ad4Ad5Ad6Ad7Ad8Ad9Ae0Ae1Ae2Ae3Ae4Ae5Ae6Ae7Ae8Ae9Af0Af1Af2Af3Af4Af5Af6Af7Af8Af9Ag0Ag1Ag2Ag3Ag4Ag5Ag6Ag7Ag8Ag9Ah0Ah1Ah2Ah3Ah4Ah5Ah6Ah7Ah8Ah9Ai0Ai1Ai2Ai3Ai4Ai5Ai6Ai7Ai8Ai9Aj0Aj1Aj2Aj3Aj4Aj5Aj6Aj7Aj8Aj9Ak0Ak1Ak2Ak3Ak4Ak5Ak6Ak7Ak8Ak9Al0Al1Al2Al3Al4Al5Al6Al7Al8Al9Am0Am1Am2Am3Am4Am5Am6Am7Am8Am9An0An1An2An3An4An5An6An7An8An9Ao0Ao1Ao2Ao3Ao4Ao5Ao6Ao7Ao8Ao9Ap0Ap1Ap2Ap3Ap4Ap5Ap6Ap7Ap8Ap9Aq0Aq1Aq2Aq3Aq4Aq5Aq6Aq7Aq8Aq9Ar0Ar1Ar2Ar3Ar4Ar5Ar6Ar7Ar8Ar9As0As1As2As3As4As5As6As7As8As9At0At1At2At3At4At5At6At7At8At9Au0Au1Au2Au3Au4Au5Au6Au7Au8Au9Av0Av1Av2Av3Av4Av5Av6Av7Av8Av9Aw0Aw1Aw2Aw3Aw4Aw5Aw6Aw7Aw8Aw9Ax0Ax1Ax2Ax3Ax4Ax5Ax6Ax7Ax8Ax9Ay0Ay1Ay2Ay3Ay4Ay5Ay6Ay7Ay8Ay9Az0Az1Az2Az3Az4Az5Az6Az7Az8Az9Ba0Ba1Ba2Ba3Ba4Ba5Ba6Ba7Ba8Ba9Bb0Bb1Bb2Bb3Bb4Bb5Bb6Bb7Bb8Bb9Bc0Bc1Bc2Bc3Bc4Bc5Bc6Bc7Bc8Bc9Bd0Bd1Bd2Bd3Bd4Bd5Bd6Bd7Bd8Bd9Be0Be1Be2Be3Be4Be5Be6Be7Be8Be9Bf0Bf1Bf2Bf3Bf4Bf5Bf6Bf7Bf8Bf9Bg0Bg1Bg2Bg3Bg4Bg5Bg6Bg7Bg8Bg9Bh0Bh1Bh2Bh3Bh4Bh5Bh6Bh7Bh8Bh9Bi0Bi1Bi2Bi3Bi4Bi5Bi6Bi7Bi8Bi9Bj0Bj1Bj2Bj3Bj4Bj5Bj6Bj7Bj8Bj9Bk0Bk1Bk2Bk3Bk4Bk5Bk6Bk7Bk8Bk9Bl0Bl1Bl2Bl3Bl4Bl5Bl6Bl7Bl8Bl9Bm0Bm1Bm2Bm3Bm4Bm5Bm6Bm7Bm8Bm9Bn0Bn1Bn2Bn3Bn4Bn5Bn6Bn7Bn8Bn9Bo0Bo1Bo2Bo3Bo4Bo5Bo6Bo7Bo8Bo9Bp0Bp1Bp2Bp3Bp4Bp5Bp6Bp7Bp8Bp9Bq0Bq1Bq2Bq3Bq4Bq5Bq6Bq7Bq8Bq9Br0Br1Br2Br3Br4Br5Br6Br7Br8Br9Bs0Bs1Bs2Bs3Bs4Bs5Bs6Bs7Bs8Bs9Bt0Bt1Bt2Bt3Bt4Bt5Bt6Bt7Bt8Bt9Bu0Bu1Bu2Bu3Bu4Bu5Bu6Bu7Bu8Bu9Bv0Bv1Bv2Bv3Bv4Bv5Bv6Bv7Bv8Bv9Bw0Bw1Bw2Bw3Bw4Bw5Bw6Bw7Bw8Bw9Bx0Bx1Bx2Bx3Bx4Bx5Bx6Bx7Bx8Bx9By0By1By2By3By4By5By6By7By8By9Bz0Bz1Bz2Bz3Bz4Bz5Bz6Bz7Bz8Bz9Ca0Ca1Ca2Ca3Ca4Ca5Ca6Ca7Ca8Ca9Cb0Cb1Cb2Cb3Cb4Cb5Cb6Cb7Cb8Cb9Cc0Cc1Cc2Cc3Cc4Cc5Cc6Cc7Cc8Cc9Cd0Cd1Cd2Cd3Cd4Cd5Cd6Cd7Cd8Cd9Ce0Ce1Ce2Ce3Ce4Ce5Ce6Ce7Ce8Ce9Cf0Cf1Cf2Cf3Cf4Cf5Cf6Cf7Cf8Cf9Cg0Cg1Cg2Cg3Cg4Cg5Cg6Cg7Cg8Cg9Ch0Ch1Ch2Ch3Ch4Ch5Ch6Ch7Ch8Ch9Ci0Ci1Ci2Ci3Ci4Ci5Ci6Ci7Ci8Ci9Cj0Cj1Cj2Cj3Cj4Cj5Cj6Cj7Cj8Cj9Ck0Ck1Ck2Ck3Ck4Ck5Ck6Ck7Ck8Ck9Cl0Cl1Cl2Cl3Cl4Cl5Cl6Cl7Cl8Cl9Cm0Cm1Cm2Cm3Cm4Cm5Cm6Cm7Cm8Cm9Cn0Cn1Cn2Cn3Cn4Cn5Cn6Cn7Cn8Cn9Co0Co1Co2Co3Co4Co5Co6Co7Co8Co9Cp0Cp1Cp2Cp3Cp4Cp5Cp6Cp7Cp8Cp9Cq0Cq1Cq2Cq3Cq4Cq5Cq6Cq7Cq8Cq9Cr0Cr1Cr2Cr3Cr4Cr5Cr6Cr7Cr8Cr9Cs0Cs1Cs2Cs3Cs4Cs5Cs6Cs7Cs8Cs9Ct0Ct1Ct2Ct3Ct4Ct5Ct6Ct7Ct8Ct9Cu0Cu1Cu2Cu3Cu4Cu5Cu6Cu7Cu8Cu9Cv0Cv1Cv2Cv3Cv4Cv5Cv6Cv7Cv8Cv9Cw0Cw1Cw2Cw3Cw4Cw5Cw6Cw7Cw8Cw9Cx0Cx1Cx2Cx3Cx4Cx5Cx6Cx7Cx8Cx9Cy0Cy1Cy2Cy3Cy4Cy5Cy6Cy7Cy8Cy9Cz0Cz1Cz2Cz3Cz4Cz5Cz6Cz7Cz8Cz9Da0Da1Da2Da3Da4Da5Da6Da7Da8Da9Db0Db1Db2Db3Db4Db5Db6Db7Db8Db9Dc0Dc1Dc2Dc3Dc4Dc5Dc6Dc7Dc8Dc9Dd0Dd1Dd2Dd3Dd4Dd5Dd6Dd7Dd8Dd9De0De1De2De3De4De5De6De7De8De9Df0Df1Df2Df3Df4Df5Df6Df7Df8Df9Dg0Dg1Dg2Dg3Dg4Dg5Dg6Dg7Dg8Dg9Dh0Dh1Dh2Dh3Dh4Dh5Dh6Dh7Dh8Dh9Di0Di1Di2Di3Di4Di5Di6Di7Di8Di9Dj0Dj1Dj2Dj3Dj4Dj5Dj6Dj7Dj8Dj9Dk0Dk1Dk2Dk3Dk4Dk5Dk6Dk7Dk8Dk9Dl0Dl1Dl2Dl3Dl4Dl5Dl6Dl7Dl8Dl9Dm0Dm1Dm2Dm3Dm4Dm5Dm6Dm7Dm8Dm9Dn0Dn1Dn2Dn3Dn4Dn5Dn6Dn7Dn8Dn9Do0Do1Do2Do3Do4Do5Do6Do7Do8Do9Dp0Dp1Dp2Dp3Dp4Dp5Dp6Dp7Dp8Dp9Dq0Dq1Dq2Dq3Dq4Dq5Dq6Dq7Dq8Dq9Dr0Dr1Dr2Dr3Dr4Dr5Dr6Dr7Dr8Dr9Ds0Ds1Ds2Ds3Ds4Ds5Ds6Ds7Ds8Ds9Dt0Dt1Dt2Dt3Dt4Dt5Dt6Dt7Dt8Dt9Du0Du1Du2Du3Du4Du5Du6Du7Du8Du9Dv0Dv1Dv2Dv3Dv4Dv5Dv6Dv7Dv8Dv9Dw0Dw1Dw2Dw3Dw4Dw5Dw6Dw7Dw8Dw9Dx0Dx1Dx2Dx3Dx4Dx5Dx6Dx7Dx8Dx9Dy0Dy1Dy2Dy3Dy4Dy5Dy6Dy7Dy8Dy9Dz0Dz1Dz2Dz3Dz4Dz5Dz6Dz7Dz8Dz9Ea0Ea1Ea2Ea3Ea4Ea5Ea6Ea7Ea8Ea9Eb0Eb1Eb2Eb3Eb4Eb5Eb6Eb7Eb8Eb9Ec0Ec1Ec2Ec3Ec4Ec5Ec6Ec7Ec8Ec9Ed0Ed1Ed2Ed3Ed4Ed5Ed6Ed7Ed8Ed9Ee0Ee1Ee2Ee3Ee4Ee5Ee6Ee7Ee8Ee9Ef0Ef1Ef2Ef3Ef4Ef5Ef6Ef7Ef8Ef9Eg0Eg1Eg2Eg3Eg4Eg5Eg6Eg7Eg8Eg9Eh0Eh1Eh2Eh3Eh4Eh5Eh6Eh7Eh8Eh9Ei0Ei1Ei2Ei3Ei4Ei5Ei6Ei7Ei8Ei9Ej0Ej1Ej2Ej3Ej4Ej5Ej6Ej7Ej8Ej9Ek0Ek1Ek2Ek3Ek4Ek5Ek6Ek7Ek8Ek9El0El1El2El3El4El5El6El7El8El9Em0Em1Em2Em3Em4Em5Em6Em7Em8Em9En0En1En2En3En4En5En6En7En8En9Eo0Eo1Eo2Eo3Eo4Eo5Eo6Eo7Eo8Eo9Ep0Ep1Ep2Ep3Ep4Ep5Ep6Ep7Ep8Ep9Eq0Eq1Eq2Eq3Eq4Eq5Eq6Eq7Eq8Eq9Er0Er1Er2Er3Er4Er5Er6Er7Er8Er9Es0Es1Es2Es3Es4Es5Es6Es7Es8Es9Et0Et1Et2Et3Et4Et5Et6Et7Et8Et9Eu0Eu1Eu2Eu3Eu4Eu5Eu6Eu7Eu8Eu9Ev0Ev1Ev2Ev3Ev4Ev5Ev6Ev7Ev8Ev9Ew0Ew1Ew2Ew3Ew4Ew5Ew6Ew7Ew8Ew9Ex0Ex1Ex2Ex3Ex4Ex5Ex6Ex7Ex8Ex9Ey0Ey1Ey2Ey3Ey4Ey5Ey6Ey7Ey8Ey9Ez0Ez1Ez2Ez3Ez4Ez5Ez6Ez7Ez8Ez9Fa0Fa1Fa2Fa3Fa4Fa5Fa6Fa7Fa8Fa9Fb0Fb1Fb2Fb3Fb4Fb5Fb6Fb7Fb8Fb9Fc0Fc1Fc2Fc3Fc4Fc5Fc6Fc7Fc8Fc9Fd0Fd1Fd2Fd3Fd4Fd5Fd6Fd7Fd8Fd9Fe0Fe1Fe2Fe3Fe4Fe5Fe6Fe7Fe8Fe9Ff0Ff1Ff2Ff3Ff4Ff5Ff6Ff7Ff8Ff9Fg0Fg1Fg2Fg3Fg4Fg5Fg6Fg7Fg8Fg9Fh0Fh1Fh2Fh3Fh4Fh5Fh6Fh7Fh8Fh9Fi0Fi1Fi2Fi3Fi4Fi5Fi6Fi7Fi8Fi9Fj0Fj1Fj2Fj3Fj4Fj5Fj6Fj7Fj8Fj9Fk0Fk1Fk2Fk3Fk4Fk5Fk6Fk7Fk8Fk9Fl0Fl1Fl2Fl3Fl4Fl5Fl6Fl7Fl8Fl9Fm0Fm1Fm2Fm3Fm4Fm5Fm6Fm7Fm8Fm9Fn0Fn1Fn2Fn3Fn4Fn5Fn6Fn7Fn8Fn9Fo0Fo1Fo2Fo3Fo4Fo5Fo6Fo7Fo8Fo9Fp0Fp1Fp2Fp3Fp4Fp5Fp6Fp7Fp8Fp9Fq0Fq1Fq2Fq3Fq4Fq5Fq6Fq7Fq8Fq9Fr0Fr1Fr2Fr3Fr4Fr5Fr6Fr7Fr8Fr9Fs0Fs1Fs2Fs3Fs4Fs5Fs6Fs7Fs8Fs9Ft0Ft1Ft2Ft3Ft4Ft5Ft6Ft7Ft8Ft9Fu0Fu1Fu2Fu3Fu4Fu5Fu6Fu7Fu8Fu9Fv0Fv1Fv2Fv3Fv4Fv5Fv6Fv7Fv8Fv9Fw0Fw1Fw2Fw3Fw4Fw5Fw6Fw7Fw8Fw9Fx0Fx1Fx2Fx3Fx4Fx5Fx6Fx7Fx8Fx9Fy0Fy1Fy2Fy3Fy4Fy5Fy6Fy7Fy8Fy9Fz0Fz1Fz2Fz3Fz4Fz5Fz6Fz7Fz8Fz9Ga0Ga1Ga2Ga3Ga4Ga5Ga6Ga7Ga8Ga9Gb0Gb1Gb2Gb3Gb4Gb5Gb6Gb7Gb8Gb9Gc0Gc1Gc2Gc3Gc4Gc5Gc6Gc7Gc8Gc9Gd0Gd1Gd2Gd3Gd4Gd5Gd6Gd7Gd8Gd9Ge0Ge1Ge2Ge3Ge4Ge5Ge6Ge7Ge8Ge9Gf0Gf1Gf2Gf3Gf4Gf5Gf6Gf7Gf8Gf9Gg0Gg1Gg2Gg3Gg4Gg5Gg6Gg7Gg8Gg9Gh0Gh1Gh2Gh3Gh4Gh5Gh6Gh7Gh8Gh9Gi0Gi1Gi2Gi3Gi4Gi5Gi6Gi7Gi8Gi9Gj0Gj1Gj2Gj3Gj4Gj5Gj6Gj7Gj8Gj9Gk0Gk1Gk2Gk3Gk4Gk5Gk6Gk7Gk8Gk9Gl0Gl1Gl2Gl3Gl4Gl5Gl6Gl7Gl8Gl9Gm0Gm1Gm2Gm3Gm4Gm5Gm6Gm7Gm8Gm9Gn0Gn1Gn2Gn3Gn4Gn5Gn6Gn7Gn8Gn9Go0Go1Go2Go3Go4Go5Go6Go7Go8Go9Gp0Gp1Gp2Gp3Gp4Gp5Gp6Gp7Gp8Gp9Gq0Gq1Gq2Gq3Gq4Gq5Gq6Gq7Gq8Gq9Gr0Gr1Gr2Gr3Gr4Gr5Gr6Gr7Gr8Gr9Gs0Gs1Gs2Gs3Gs4Gs5Gs6Gs7Gs8Gs9Gt0Gt1Gt2Gt3Gt4Gt5Gt6Gt7Gt8Gt9Gu0Gu1Gu2Gu3Gu4Gu5Gu6Gu7Gu8Gu9Gv0Gv1Gv2Gv3Gv4Gv5Gv6Gv7Gv8Gv9Gw0Gw1Gw2Gw3Gw4Gw5Gw6Gw7Gw8Gw9Gx0Gx1Gx2Gx3Gx4Gx5Gx6Gx7Gx8Gx9Gy0Gy1Gy2Gy3Gy4Gy5Gy6Gy7Gy8Gy9Gz0Gz1Gz2Gz3Gz4Gz5Gz6Gz7Gz8Gz9Ha0Ha1Ha2Ha3Ha4Ha5Ha6Ha7Ha8Ha9Hb0Hb1Hb2Hb3Hb4Hb5Hb6Hb7Hb8Hb9Hc0Hc1Hc2Hc3Hc4Hc5Hc6Hc7Hc8Hc9Hd0Hd1Hd2Hd3Hd4Hd5Hd6Hd7Hd8Hd9He0He1He2He3He4He5He6He7He8He9Hf0Hf1Hf2Hf3Hf4Hf5Hf6Hf7Hf8Hf9Hg0Hg1Hg2Hg3Hg4Hg5Hg6Hg7Hg8Hg9Hh0Hh1Hh2Hh3Hh4Hh5Hh6Hh7Hh8Hh9Hi0Hi1Hi2Hi3Hi4Hi5Hi6Hi7Hi8Hi9Hj0Hj1Hj2Hj3Hj4Hj5Hj6Hj7Hj8Hj9Hk0Hk1Hk2Hk3Hk4Hk5Hk6Hk7Hk8Hk9Hl0Hl1Hl2Hl3Hl4Hl5Hl6Hl7Hl8Hl9Hm0Hm1Hm2Hm3Hm4Hm5Hm6Hm7Hm8Hm9Hn0Hn1Hn2Hn3Hn4Hn5Hn6Hn7Hn8Hn9Ho0Ho1Ho2Ho3Ho4Ho5Ho6Ho7Ho8Ho9Hp0Hp1Hp2Hp3Hp4Hp5Hp6Hp7Hp8Hp9Hq0Hq1Hq2Hq3Hq4Hq5Hq6Hq7Hq8Hq9Hr0Hr1Hr2Hr3Hr4Hr5Hr6Hr7Hr8Hr9Hs0Hs1Hs2Hs3Hs4Hs5Hs6Hs7Hs8Hs9Ht0Ht1Ht2Ht3Ht4Ht5Ht6Ht7Ht8Ht9Hu0Hu1Hu2Hu3Hu4Hu5Hu6Hu7Hu8Hu9Hv0Hv1Hv2Hv3Hv4Hv5Hv6Hv7Hv8Hv9Hw0Hw1Hw2Hw3Hw4Hw5Hw6Hw7Hw8Hw9Hx0Hx1Hx2Hx3Hx4Hx5Hx6Hx7Hx8Hx9Hy0Hy1Hy2Hy3Hy4Hy5Hy6Hy7Hy8Hy9Hz0Hz1Hz2Hz3Hz4Hz5Hz6Hz7Hz8Hz9Ia0Ia1Ia2Ia3Ia4Ia5Ia6Ia7Ia8Ia9Ib0Ib1Ib2Ib3Ib4Ib5Ib6Ib7Ib8Ib9Ic0Ic1Ic2Ic3Ic4Ic5Ic6Ic7Ic8Ic9Id0Id1Id2Id3Id4Id5Id6Id7Id8Id9Ie0Ie1Ie2Ie3Ie4Ie5Ie6Ie7Ie8Ie9If0If1If2If3If4If5If6If7If8If9Ig0Ig1Ig2Ig3Ig4Ig5Ig6Ig7Ig8Ig9Ih0Ih1Ih2Ih3Ih4Ih5Ih6Ih7Ih8Ih9Ii0Ii1Ii2Ii3Ii4Ii5Ii6Ii7Ii8Ii9Ij0Ij1Ij2Ij3Ij4Ij5Ij6Ij7Ij8Ij9Ik0Ik1Ik2Ik3Ik4Ik5Ik6Ik7Ik8Ik9Il0Il1Il2Il3Il4Il5Il6Il7Il8Il9Im0Im1Im2Im3Im4Im5Im6Im7Im8Im9In0In1In2In3In4In5In6In7In8In9Io0Io1Io2Io3Io4Io5Io6Io7Io8Io9Ip0Ip1Ip2Ip3Ip4Ip5Ip6Ip7Ip8Ip9Iq0Iq1Iq2Iq3Iq4Iq5Iq6Iq7Iq8Iq9Ir0Ir1Ir2Ir3Ir4Ir5Ir6Ir7Ir8Ir9Is0Is1Is2Is3Is4Is5Is6Is7Is8Is9It0It1It2It3It4It5It6It7It8It9Iu0Iu1Iu2Iu3Iu4Iu5Iu6Iu7Iu8Iu9Iv0Iv1Iv2Iv3Iv4Iv5Iv6Iv7Iv8Iv9Iw0Iw1Iw2Iw3Iw4Iw5Iw6Iw7Iw8Iw9Ix0Ix1Ix2Ix3Ix4Ix5Ix6Ix7Ix8Ix9Iy0Iy1Iy2Iy3Iy4Iy5Iy6Iy7Iy8Iy9Iz0Iz1Iz2Iz3Iz4Iz5Iz6Iz7Iz8Iz9Ja0Ja1Ja2Ja3Ja4Ja5Ja6Ja7Ja8Ja9Jb0Jb1Jb2Jb3Jb4Jb5Jb6Jb7Jb8Jb9Jc0Jc1Jc2Jc3Jc4Jc5Jc6Jc7Jc8Jc9Jd0Jd1Jd2Jd3Jd4Jd5Jd6Jd7Jd8Jd9Je0Je1Je2Je3Je4Je5Je6Je7Je8Je9Jf0Jf1Jf2Jf3Jf4Jf5Jf6Jf7Jf8Jf9Jg0Jg1Jg2Jg3Jg4Jg5Jg6Jg7Jg8Jg9Jh0Jh1Jh2Jh3Jh4Jh5Jh6Jh7Jh8Jh9Ji0Ji1Ji2Ji3Ji4Ji5Ji6Ji7Ji8Ji9Jj0Jj1Jj2Jj3Jj4Jj5Jj6Jj7Jj8Jj9Jk0Jk1Jk2Jk3Jk4Jk5Jk6Jk7Jk8Jk9Jl0Jl1Jl2Jl3Jl4Jl5Jl6Jl7Jl8Jl9Jm0Jm1Jm2Jm3Jm4Jm5Jm6Jm7Jm8Jm9Jn0Jn1Jn2Jn3Jn4Jn5Jn6Jn7Jn8Jn9Jo0Jo1Jo2Jo3Jo4Jo5Jo6Jo7Jo8Jo9Jp0Jp1Jp2Jp3Jp4Jp5Jp6Jp7Jp8Jp9Jq0Jq1Jq2Jq3Jq4Jq5Jq6Jq7Jq8Jq9Jr0Jr1Jr2Jr3Jr4Jr5Jr6Jr7Jr8Jr9Js0Js1Js2Js3Js4Js5Js6Js7Js8Js9Jt0Jt1Jt2Jt3Jt4Jt5Jt6Jt7Jt8Jt9Ju0Ju1Ju2Ju3Ju4Ju5Ju6Ju7Ju8Ju9Jv0Jv1Jv2Jv3Jv4Jv5Jv6Jv7Jv8Jv9Jw0Jw1Jw2Jw3Jw4Jw5Jw6Jw7Jw8Jw9Jx0Jx1Jx2Jx3Jx4Jx5Jx6Jx7Jx8Jx9Jy0Jy1Jy2Jy3Jy4Jy5Jy6Jy7Jy8Jy9Jz0Jz1Jz2Jz3Jz4Jz5Jz6Jz7Jz8Jz9Ka0Ka1Ka2Ka3Ka4Ka5Ka6Ka7Ka8Ka9Kb0Kb1Kb2Kb3Kb4Kb5Kb6Kb7Kb8Kb9Kc0Kc1Kc2Kc3Kc4Kc5Kc6Kc7Kc8Kc9Kd0Kd1Kd2Kd3Kd4Kd5Kd6Kd7Kd8Kd9Ke0Ke1Ke2Ke3Ke4Ke5Ke6Ke7Ke8Ke9Kf0Kf1Kf2Kf3Kf4Kf5Kf6Kf7Kf8Kf9Kg0Kg1Kg2Kg3Kg4Kg5Kg6Kg7Kg8Kg9Kh0Kh1Kh2Kh3Kh4Kh5Kh6Kh7Kh8Kh9Ki0Ki1Ki2Ki3Ki4Ki5Ki6Ki7Ki8Ki9Kj0Kj1Kj2Kj3Kj4Kj5Kj6Kj7Kj8Kj9Kk0Kk1Kk2Kk3Kk4Kk5Kk6Kk7Kk8Kk9Kl0Kl1Kl2Kl3Kl4Kl5Kl6Kl7Kl8Kl9Km0Km1Km2Km3Km4Km5Km6Km7Km8Km9Kn0Kn1Kn2Kn3Kn4Kn5Kn6Kn7Kn8Kn9Ko0Ko1Ko2Ko3Ko4Ko5Ko6Ko7Ko8Ko9Kp0Kp1Kp2Kp3Kp4Kp5Kp6Kp7Kp8Kp9Kq0Kq1Kq2Kq3Kq4Kq5Kq6Kq7Kq8Kq9Kr0Kr1Kr2Kr3Kr4Kr5Kr6Kr7Kr8Kr9Ks0Ks1Ks2Ks3Ks4Ks5Ks6Ks7Ks8Ks9Kt0Kt1Kt2Kt3Kt4Kt5Kt6Kt7Kt8Kt9Ku0Ku1Ku2Ku3Ku4Ku5Ku6Ku7Ku8Ku9Kv0Kv1Kv2Kv3Kv4Kv5Kv6Kv7Kv8Kv9Kw0Kw1Kw2Kw3Kw4Kw5Kw6Kw7Kw8Kw9Kx0Kx1Kx2Kx3Kx4Kx5Kx6Kx7Kx8Kx9Ky0Ky1Ky2Ky3Ky4Ky5Ky6Ky7Ky8Ky9Kz0Kz1Kz2Kz3Kz4Kz5Kz6Kz7Kz8Kz9La0La1La2La3La4La5La6La7La8La9Lb0Lb1Lb2Lb3Lb4Lb5Lb6Lb7Lb8Lb9Lc0Lc1Lc2Lc3Lc4Lc5Lc6Lc7Lc8Lc9Ld0Ld1Ld2Ld3Ld4Ld5Ld6Ld7Ld8Ld9Le0Le1Le2Le3Le4Le5Le6Le7Le8Le9Lf0Lf1Lf2Lf3Lf4Lf5Lf6Lf7Lf8Lf9Lg0Lg1Lg2Lg3Lg4Lg5Lg6Lg7Lg8Lg9Lh0Lh1Lh2Lh3Lh4Lh5Lh6Lh7Lh8Lh9Li0Li1Li2Li3Li4Li5Li6Li7Li8Li9Lj0Lj1Lj2Lj3Lj4Lj5Lj6Lj7Lj8Lj9Lk0Lk1Lk2Lk3Lk4Lk5Lk6Lk7Lk8Lk9Ll0Ll1Ll2Ll3Ll4Ll5Ll6Ll7Ll8Ll9Lm0Lm1Lm2Lm3Lm4Lm5Lm6Lm7Lm8Lm9Ln0Ln1Ln2Ln3Ln4Ln5Ln6Ln7Ln8Ln9Lo0Lo1Lo2Lo3Lo4Lo5Lo6Lo7Lo8Lo9Lp0Lp1Lp2Lp3Lp4Lp5Lp6Lp7Lp8Lp9Lq0Lq1Lq2Lq3Lq4Lq5Lq6Lq7Lq8Lq9Lr0Lr1Lr2Lr3Lr4Lr5Lr6Lr7Lr8Lr9Ls0Ls1Ls2Ls3Ls4Ls5Ls6Ls7Ls8Ls9Lt0Lt1Lt2Lt3Lt4Lt5Lt6Lt7Lt8Lt9Lu0Lu1Lu2Lu3Lu4Lu5Lu6Lu7Lu8Lu9Lv0Lv1Lv2Lv3Lv4Lv5Lv6Lv7Lv8Lv9Lw0Lw1Lw2Lw3Lw4Lw5Lw6Lw7Lw8Lw9Lx0Lx1Lx2Lx3Lx4Lx5Lx6Lx7Lx8Lx9Ly0Ly1Ly2Ly3Ly4Ly5Ly6Ly7Ly8Ly9Lz0Lz1Lz2Lz3Lz4Lz5Lz6Lz7Lz8Lz9Ma0Ma1Ma2Ma3Ma4Ma5Ma6Ma7Ma8Ma9Mb0Mb1Mb2Mb3Mb4Mb5Mb6Mb7Mb8Mb9Mc0Mc1Mc2Mc3Mc4Mc5Mc6Mc7Mc8Mc9Md0Md1Md2Md3Md4Md5Md6Md7Md8Md9Me0Me1Me2Me3Me4Me5Me6Me7Me8Me9Mf0Mf1Mf2Mf3Mf4Mf5Mf6Mf7Mf8Mf9Mg0Mg1Mg2Mg3Mg4Mg5Mg6Mg7Mg8Mg9Mh0Mh1Mh2Mh3Mh4Mh5Mh6Mh7Mh8Mh9Mi0Mi1Mi2Mi3Mi4Mi5Mi6Mi7Mi8Mi9Mj0Mj1Mj2Mj3Mj4Mj5Mj6Mj7Mj8Mj9Mk0Mk1Mk2Mk3Mk4Mk5Mk6Mk7Mk8Mk9Ml0Ml1Ml2Ml3Ml4Ml5Ml6Ml7Ml8Ml9Mm0Mm1Mm2Mm3Mm4Mm5Mm6Mm7Mm8Mm9Mn0Mn1Mn2Mn3Mn4Mn5Mn6Mn7Mn8Mn9Mo0Mo1Mo2Mo3Mo4Mo5Mo6Mo7Mo8Mo9Mp0Mp1Mp2Mp3Mp4Mp5Mp6Mp7Mp8Mp9Mq0Mq1Mq2Mq3Mq4Mq5Mq6Mq7Mq8Mq9Mr0Mr1Mr2Mr3Mr4Mr5Mr6Mr7Mr8Mr9Ms0Ms1Ms2Ms3Ms4Ms5Ms6Ms7Ms8Ms9Mt0Mt1Mt2Mt3Mt4Mt5Mt6Mt7Mt8Mt9Mu0Mu1Mu2Mu3Mu4Mu5Mu6Mu7Mu8Mu9Mv0Mv1Mv2Mv3Mv4Mv5Mv6Mv7Mv8Mv9Mw0Mw1Mw2Mw3Mw4Mw5Mw6Mw7Mw8Mw9Mx0Mx1Mx2Mx3Mx4Mx5Mx6Mx7Mx8Mx9My0My1My2My3My4My5My6My7My8My9Mz0Mz1Mz2Mz3Mz4Mz5Mz6Mz7Mz8Mz9Na0Na1Na2Na3Na4Na5Na6Na7Na8Na9Nb0Nb1Nb2Nb3Nb4Nb5Nb6Nb7Nb8Nb9Nc0Nc1Nc2Nc3Nc4Nc5Nc6Nc7Nc8Nc9Nd0Nd1Nd2Nd3Nd4Nd5Nd6Nd7Nd8Nd9Ne0Ne1Ne2Ne3Ne4Ne5Ne6Ne7Ne8Ne9Nf0Nf1Nf2Nf3Nf4Nf5Nf6Nf7Nf8Nf9Ng0Ng1Ng2Ng3Ng4Ng5Ng6Ng7Ng8Ng9Nh0Nh1Nh2Nh3Nh4Nh5Nh6Nh7Nh8Nh9Ni0Ni1Ni2Ni3Ni4Ni5Ni6Ni7Ni8Ni9Nj0Nj1Nj2Nj3Nj4Nj5Nj6Nj7Nj8Nj9Nk0Nk1Nk2Nk3Nk4Nk5Nk6Nk7Nk8Nk9Nl0Nl1Nl2Nl3Nl4Nl5Nl6Nl7Nl8Nl9Nm0Nm1Nm2Nm3Nm4Nm5Nm6Nm7Nm8Nm9Nn0Nn1Nn2Nn3Nn4Nn5Nn6Nn7Nn8Nn9No0No1No2No3No4No5No6No7No8No9Np0Np1Np2Np3Np4Np5Np6Np7Np8Np9Nq0Nq1Nq2Nq3Nq4Nq5Nq6Nq7Nq8Nq9Nr0Nr1Nr2Nr3Nr4Nr5Nr6Nr7Nr8Nr9Ns0Ns1Ns2Ns3Ns4Ns5Ns6Ns7Ns8Ns9Nt0Nt1Nt2Nt3Nt4Nt5Nt6Nt7Nt8Nt9Nu0Nu1Nu2Nu3Nu4Nu5Nu6Nu7Nu8Nu9Nv0Nv1Nv2Nv3Nv4Nv5Nv6Nv7Nv8Nv9Nw0Nw1Nw2Nw3Nw4Nw5Nw6Nw7Nw8Nw9Nx0Nx1Nx2Nx3Nx4Nx5Nx6Nx7Nx8Nx9Ny0Ny1Ny2Ny3Ny4Ny5Ny6Ny7Ny8Ny9Nz0Nz1Nz2Nz3Nz4Nz5Nz6Nz7Nz8Nz9Oa0Oa1Oa2Oa3Oa4Oa5Oa6Oa7Oa8Oa9Ob0Ob1Ob2Ob3Ob4Ob5Ob6Ob7Ob8Ob9Oc0Oc1Oc2Oc3Oc4Oc5Oc6Oc7Oc8Oc9Od0Od1Od2Od3Od4Od5Od6Od7Od8Od9Oe0Oe1Oe2Oe3Oe4Oe5Oe6Oe7Oe8Oe9Of0Of1Of2Of3Of4Of5Of6Of7Of8Of9Og0Og1Og2Og3Og4Og5Og6Og7Og8Og9Oh0Oh1Oh2Oh3Oh4Oh5Oh6Oh7Oh8Oh9Oi0Oi1Oi2Oi3Oi4Oi5Oi6Oi7Oi8Oi9Oj0Oj1Oj2Oj3Oj4Oj5Oj6Oj7Oj8Oj9Ok0Ok1Ok2Ok3Ok4Ok5Ok6Ok7Ok8Ok9Ol0Ol1Ol2Ol3Ol4Ol5Ol6Ol7Ol8Ol9Om0Om1Om2Om3Om4Om5Om6Om7Om8Om9On0On1On2On3On4On5On6On7On8On9Oo0Oo1Oo2Oo3Oo4Oo5Oo6Oo7Oo8Oo9Op0Op1Op2Op3Op4Op5Op6Op7Op8Op9Oq0Oq1Oq2Oq3Oq4Oq5Oq6Oq7Oq8Oq9Or0Or1Or2Or3Or4Or5Or6Or7Or8Or9Os0Os1Os2Os3Os4Os5Os6Os7Os8Os9Ot0Ot1Ot2Ot3Ot4Ot5Ot6Ot7Ot8Ot9Ou0Ou1Ou2Ou3Ou4Ou5Ou6Ou7Ou8Ou9Ov0Ov1Ov2Ov3Ov4Ov5Ov6Ov7Ov8Ov9Ow0Ow1Ow2Ow3Ow4Ow5Ow6Ow7Ow8Ow9Ox0Ox1Ox2Ox3Ox4Ox5Ox6Ox7Ox8Ox9Oy0Oy1Oy2Oy3Oy4Oy5Oy6Oy7Oy8Oy9Oz0Oz1Oz2Oz3Oz4Oz5Oz6Oz7Oz8Oz9Pa0Pa1Pa2Pa3Pa4Pa5Pa6Pa7Pa8Pa9Pb0Pb1Pb2Pb3Pb4Pb5Pb6Pb7Pb8Pb9Pc0Pc1Pc2Pc3Pc4Pc5Pc6Pc7Pc8Pc9Pd0Pd1Pd2Pd3Pd4Pd5Pd6Pd7Pd8Pd9Pe0Pe1Pe2Pe3Pe4Pe5Pe6Pe7Pe8Pe9Pf0Pf1Pf2Pf3Pf4Pf5Pf6Pf7Pf8Pf9Pg0Pg1Pg2Pg3Pg4Pg5Pg6Pg7Pg8Pg9Ph0Ph1Ph2Ph3Ph4Ph5Ph6Ph7Ph8Ph9Pi0Pi1Pi2Pi3Pi4Pi5Pi6Pi7Pi8Pi9Pj0Pj1Pj2Pj3Pj4Pj5Pj6Pj7Pj8Pj9Pk0Pk1Pk2Pk3Pk4Pk5Pk6Pk7Pk8Pk9Pl0Pl1Pl2Pl3Pl4Pl5Pl6Pl7Pl8Pl9Pm0Pm1Pm2Pm3Pm4Pm5Pm6Pm7Pm8Pm9Pn0Pn1Pn2Pn3Pn4Pn5Pn6Pn7Pn8Pn9Po0Po1Po2Po3Po4Po5Po6Po7Po8Po9Pp0Pp1Pp2Pp3Pp4Pp5Pp6Pp7Pp8Pp9Pq0Pq1Pq2Pq3Pq4Pq5Pq6Pq7Pq8Pq9Pr0Pr1Pr2Pr3Pr4Pr5Pr6Pr7Pr8Pr9Ps0Ps1Ps2Ps3Ps4Ps5Ps6Ps7Ps8Ps9Pt0Pt1Pt2Pt3Pt4Pt5Pt6Pt7Pt8Pt9Pu0Pu1Pu2Pu3Pu4Pu5Pu6Pu7Pu8Pu9Pv0Pv1Pv2Pv3Pv4Pv5Pv6Pv7Pv8Pv9Pw0Pw1Pw2Pw3Pw4Pw5Pw6Pw7Pw8Pw9Px0Px1Px2Px3Px4Px5Px6Px7Px8Px9Py0Py1Py2Py3Py4Py5Py6Py7Py8Py9Pz0Pz1Pz2Pz3Pz4Pz5Pz6Pz7Pz8Pz9Qa0Qa1Qa2Qa3Qa4Qa5Qa6Qa7Qa8Qa9Qb0Qb1Qb2Qb3Qb4Qb5Qb6Qb7Qb8Qb9Qc0Qc1Qc2Qc3Qc4Qc5Qc6Qc7Qc8Qc9Qd0Qd1Qd2Qd3Qd4Qd5Qd6Qd7Qd8Qd9Qe0Qe1Qe2Qe3Qe4Qe5Qe6Qe7Qe8Qe9Qf0Qf1Qf2Qf3Qf4Qf5Qf6Qf7Qf8Qf9Qg0Qg1Qg2Qg3Qg4Qg5Qg6Qg7Qg8Qg9Qh0Qh1Qh2Qh3Qh4Qh5Qh6Qh7Qh8Qh9Qi0Qi1Qi2Qi3Qi4Qi5Qi6Qi7Qi8Qi9Qj0Qj1Qj2Qj3Qj4Qj5Qj6Qj7Qj8Qj9Qk0Qk1Qk2Qk3Qk4Qk5Qk6Qk7Qk8Qk9Ql0Ql1Ql2Ql3Ql4Ql5Ql6Ql7Ql8Ql9Qm0Qm1Qm2Qm3Qm4Qm5Qm6Qm7Qm8Qm9Qn0Qn1Qn2Qn3Qn4Qn5Qn6Qn7Qn8Qn9Qo0Qo1Qo2Qo3Qo4Qo5Qo6Qo7Qo8Qo9Qp0Qp1Qp2Qp3Qp4Qp5Qp6Qp7Qp8Qp9Qq0Qq1Qq2Qq3Qq4Qq5Qq6Qq7Qq8Qq9Qr0Qr1Qr2Qr3Qr4Qr5Qr6Qr7Qr8Qr9Qs0Qs1Qs2Qs3Qs4Qs5Qs6Qs7Qs8Qs9Qt0Qt1Qt2Qt3Qt4Qt5Qt6Qt7Qt8Qt9Qu0Qu1Qu2Qu3Qu4Qu5Qu6Qu7Qu8Qu9Qv0Qv1Qv2Qv3Qv4Qv5Qv6Qv7Qv8Qv9Qw0Qw1Qw2Qw3Qw4Qw5Qw6Qw7Qw8Qw9Qx0Qx1Qx2Qx3Qx4Qx5Qx6Qx7Qx8Qx9Qy0Qy1Qy2Qy3Qy4Qy5Qy6Qy7Qy8Qy9Qz0Qz1Qz2Qz3Qz4Qz5Qz6Qz7Qz8Qz9Ra0Ra1Ra2Ra3Ra4Ra5Ra6Ra7Ra8Ra9Rb0Rb1Rb2Rb3Rb4Rb5Rb6Rb7Rb8Rb9Rc0Rc1Rc2Rc3Rc4Rc5Rc6Rc7Rc8Rc9Rd0Rd1Rd2Rd3Rd4Rd5Rd6Rd7Rd8Rd9Re0Re1Re2Re3Re4Re5Re6Re7Re8Re9Rf0Rf1Rf2Rf3Rf4Rf5Rf6Rf7Rf8Rf9Rg0Rg1Rg2Rg3Rg4Rg5Rg6Rg7Rg8Rg9Rh0Rh1Rh2Rh3Rh4Rh5Rh6Rh7Rh8Rh9Ri0Ri1Ri2Ri3Ri4Ri5Ri6Ri7Ri8Ri9Rj0Rj1Rj2Rj3Rj4Rj5Rj6Rj7Rj8Rj9Rk0Rk1Rk2Rk3Rk4Rk5Rk6Rk7Rk8Rk9Rl0Rl1Rl2Rl3Rl4Rl5Rl6Rl7Rl8Rl9Rm0Rm1Rm2Rm3Rm4Rm5Rm6Rm7Rm8Rm9Rn0Rn1Rn2Rn3Rn4Rn5Rn6Rn7Rn8Rn9Ro0Ro1Ro2Ro3Ro4Ro5Ro6Ro7Ro8Ro9Rp0Rp1Rp2Rp3Rp4Rp5Rp6Rp7Rp8Rp9Rq0Rq1Rq2Rq3Rq4Rq5Rq6Rq7Rq8Rq9Rr0Rr1Rr2Rr3Rr4Rr5Rr6Rr7Rr8Rr9Rs0Rs1Rs2Rs3Rs4Rs5Rs6Rs7Rs8Rs9Rt0Rt1Rt2Rt3Rt4Rt5Rt6Rt7Rt8Rt9Ru0Ru1Ru2Ru3Ru4Ru5Ru6Ru7Ru8Ru9Rv0Rv1Rv2Rv3Rv4Rv5Rv6Rv7Rv8Rv9Rw0Rw1Rw2Rw3Rw4Rw5Rw6Rw7Rw8Rw9Rx0Rx1Rx2Rx3Rx4Rx5Rx6Rx7Rx8Rx9Ry0Ry1Ry2Ry3Ry4Ry5Ry6Ry7Ry8Ry9Rz0Rz1Rz2Rz3Rz4Rz5Rz6Rz7Rz8Rz9Sa0Sa1Sa2Sa3Sa4Sa5Sa6Sa7Sa8Sa9Sb0Sb1Sb2Sb3Sb4Sb5Sb6Sb7Sb8Sb9Sc0Sc1Sc2Sc3Sc4Sc5Sc6Sc7Sc8Sc9Sd0Sd1Sd2Sd3Sd4Sd5Sd6Sd7Sd8Sd9Se0Se1Se2Se3Se4Se5Se6Se7Se8Se9Sf0Sf1Sf2Sf3Sf4Sf5Sf6Sf7Sf8Sf9Sg0Sg1Sg2Sg3Sg4Sg5Sg6Sg7Sg8Sg9Sh0Sh1Sh2Sh3Sh4Sh5Sh6Sh7Sh8Sh9Si0Si1Si2Si3Si4Si5Si6Si7Si8Si9Sj0Sj1Sj2Sj3Sj4Sj5Sj6Sj7Sj8Sj9Sk0Sk1Sk2Sk3Sk4Sk5Sk6Sk7Sk8Sk9Sl0Sl1Sl2Sl3Sl4Sl5Sl6Sl7Sl8Sl9Sm0Sm1Sm2Sm3Sm4Sm5Sm6Sm7Sm8Sm9Sn0Sn1Sn2Sn3Sn4Sn5Sn6Sn7Sn8Sn9So0So1So2So3So4So5So6So7So8So9Sp0Sp1Sp2Sp3Sp4Sp5Sp6Sp7Sp8Sp9Sq0Sq1Sq2Sq3Sq4Sq5Sq6Sq7Sq8Sq9Sr0Sr1Sr2Sr3Sr4Sr5Sr6Sr7Sr8Sr9Ss0Ss1Ss2Ss3Ss4Ss5Ss6Ss7Ss8Ss9St0St1St2St3St4St5St6St7St8St9Su0Su1Su2Su3Su4Su5Su6Su7Su8Su9Sv0Sv1Sv2Sv3Sv4Sv5Sv6Sv7Sv8Sv9Sw0Sw1Sw2Sw3Sw4Sw5Sw6Sw7Sw8Sw9Sx0Sx1Sx2Sx3Sx4Sx5Sx6Sx7Sx8Sx9Sy0Sy1Sy2Sy3Sy4Sy5Sy6Sy7Sy8Sy9Sz0Sz1Sz2Sz3Sz4Sz5Sz6Sz7Sz8Sz9Ta0Ta1Ta2Ta3Ta4Ta5Ta6Ta7Ta8Ta9Tb0Tb1Tb2Tb3Tb4Tb5Tb6Tb7Tb8Tb9Tc0Tc1Tc2Tc3Tc4Tc5Tc6Tc7Tc8Tc9Td0Td1Td2Td3Td4Td5Td6Td7Td8Td9Te0Te1Te2Te3Te4Te5Te6Te7Te8Te9Tf0Tf1Tf2Tf3Tf4Tf5Tf6Tf7Tf8Tf9Tg0Tg1Tg2Tg3Tg4Tg5Tg6Tg7Tg8Tg9Th0Th1Th2Th3Th4Th5Th6Th7Th8Th9Ti0Ti1Ti2Ti3Ti4Ti5Ti6Ti7Ti8Ti9Tj0Tj1Tj2Tj3Tj4Tj5Tj6Tj7Tj8Tj9Tk0Tk1Tk2Tk3Tk4Tk5Tk6Tk7Tk8Tk9Tl0Tl1Tl2Tl3Tl4Tl5Tl6Tl7Tl8Tl9Tm0Tm1Tm2Tm3Tm4Tm5Tm6Tm7Tm8Tm9Tn0Tn1Tn2Tn3Tn4Tn5Tn6Tn7Tn8Tn9To0To1To2To3To4To5To6To7To8To9Tp0Tp1Tp2Tp3Tp4Tp5Tp6Tp7Tp8Tp9Tq0Tq1Tq2Tq3Tq4Tq5Tq6Tq7Tq8Tq9Tr0Tr1Tr2Tr3Tr4Tr5Tr6Tr7Tr8Tr9Ts0Ts1Ts2Ts3Ts4Ts5Ts6Ts7Ts8Ts9Tt0Tt1Tt2Tt3Tt4Tt5Tt6Tt7Tt8Tt9Tu0Tu1Tu2Tu3Tu4Tu5Tu6Tu7Tu8Tu9Tv0Tv1Tv2Tv3Tv4Tv5Tv6Tv7Tv8Tv9Tw0Tw1Tw2Tw3Tw4Tw5Tw6Tw7Tw8Tw9Tx0Tx1Tx2Tx3Tx4Tx5Tx6Tx7Tx8Tx9Ty0Ty1Ty2Ty3Ty4Ty5Ty6Ty7Ty8Ty9Tz0Tz1Tz2Tz3Tz4Tz5Tz6Tz7Tz8Tz9Ua0Ua1Ua2Ua3Ua4Ua5Ua6Ua7Ua8Ua9Ub0Ub1Ub2Ub3Ub4Ub5Ub6Ub7Ub8Ub9Uc0Uc1Uc2Uc3Uc4Uc5Uc6Uc7Uc8Uc9Ud0Ud1Ud2Ud3Ud4Ud5Ud6Ud7Ud8Ud9Ue0Ue1Ue2Ue3Ue4Ue5Ue6Ue7Ue8Ue9Uf0Uf1Uf2Uf3Uf4Uf5Uf6Uf7Uf8Uf9Ug0Ug1Ug2Ug3Ug4Ug5Ug6Ug7Ug8Ug9Uh0Uh1Uh2Uh3Uh4Uh5Uh6Uh7Uh8Uh9Ui0Ui1Ui2Ui3Ui4Ui5Ui6Ui7Ui8Ui9Uj0Uj1Uj2Uj3Uj4Uj5Uj6Uj7Uj8Uj9Uk0Uk1Uk2Uk3Uk4Uk5Uk6Uk7Uk8Uk9Ul0Ul1Ul2Ul3Ul4Ul5Ul6Ul7Ul8Ul9Um0Um1Um2Um3Um4Um5Um6Um7Um8Um9Un0Un1Un2Un3Un4Un5Un6Un7Un8Un9Uo0Uo1Uo2Uo3Uo4Uo5Uo6Uo7Uo8Uo9Up0Up1Up2Up3Up4Up5Up6Up7Up8Up9Uq0Uq1Uq2Uq3Uq4Uq5Uq6Uq7Uq8Uq9Ur0Ur1Ur2Ur3Ur4Ur5Ur6Ur7Ur8Ur9Us0Us1Us2Us3Us4Us5Us6Us7Us8Us9Ut0Ut1Ut2Ut3Ut4Ut5Ut6Ut7Ut8Ut9Uu0Uu1Uu2Uu3Uu4Uu5Uu6Uu7Uu8Uu9Uv0Uv1Uv2Uv3Uv4Uv5Uv6Uv7Uv8Uv9Uw0Uw1Uw2Uw3Uw4Uw5Uw6Uw7Uw8Uw9Ux0Ux1Ux2Ux3Ux4Ux5Ux6Ux7Ux8Ux9Uy0Uy1Uy2Uy3Uy4Uy5Uy6Uy7Uy8Uy9Uz0Uz1Uz2Uz3Uz4Uz5Uz6Uz7Uz8Uz9Va0Va1Va2Va3Va4Va5Va6Va7Va8Va9Vb0Vb1Vb2Vb3Vb4Vb5Vb6Vb7Vb8Vb9Vc0Vc1Vc2Vc3Vc4Vc5Vc6Vc7Vc8Vc9Vd0Vd1Vd2Vd3Vd4Vd5Vd6Vd7Vd8Vd9Ve0Ve1Ve2Ve3Ve4Ve5Ve6Ve7Ve8Ve9Vf0Vf1Vf2Vf3Vf4Vf5Vf6Vf7Vf8Vf9Vg0Vg1Vg2Vg3Vg4Vg5Vg6Vg7Vg8Vg9Vh0Vh1Vh2Vh3Vh4Vh5Vh6Vh7Vh8Vh9Vi0Vi1Vi2Vi3Vi4Vi5Vi6Vi7Vi8Vi9Vj0Vj1Vj2Vj3Vj4Vj5Vj6Vj7Vj8Vj9Vk0Vk1Vk2Vk3Vk4Vk5Vk6Vk7Vk8Vk9Vl0Vl1Vl2Vl3Vl4Vl5Vl6Vl7Vl8Vl9Vm0Vm1Vm2Vm3Vm4Vm5Vm6Vm7Vm8Vm9Vn0Vn1Vn2Vn3Vn4Vn5Vn6Vn7Vn8Vn9Vo0Vo1Vo2Vo3Vo4Vo5Vo6Vo7Vo8Vo9Vp0Vp1Vp2Vp3Vp4Vp5Vp6Vp7Vp8Vp9Vq0Vq1Vq2Vq3Vq4Vq5Vq6Vq7Vq8Vq9Vr0Vr1Vr2Vr3Vr4Vr5Vr6Vr7Vr8Vr9Vs0Vs1Vs2Vs3Vs4Vs5Vs6Vs7Vs8Vs9Vt0Vt1Vt2Vt3Vt4Vt5Vt6Vt7Vt8Vt9Vu0Vu1Vu2Vu3Vu4Vu5Vu6Vu7Vu8Vu9Vv0Vv1Vv2Vv3Vv4Vv5Vv6Vv7Vv8Vv9Vw0Vw1Vw2Vw3Vw4Vw5Vw6Vw7Vw8Vw9Vx0Vx1Vx2Vx3Vx4Vx5Vx6Vx7Vx8Vx9Vy0Vy1Vy2Vy3Vy4Vy5Vy6Vy7Vy8Vy9Vz0Vz1Vz2Vz3Vz4Vz5Vz6Vz7Vz8Vz9Wa0Wa1Wa2Wa3Wa4Wa5Wa6Wa7Wa8Wa9Wb0Wb1Wb2Wb3Wb4Wb5Wb6Wb7Wb8Wb9Wc0Wc1Wc2Wc3Wc4Wc5Wc6Wc7Wc8Wc9Wd0Wd1Wd2Wd3Wd4Wd5Wd6Wd7Wd8Wd9We0We1We2We3We4We5We6We7We8We9Wf0Wf1Wf2Wf3Wf4Wf5Wf6Wf7Wf8Wf9Wg0Wg1Wg2Wg3Wg4Wg5Wg6Wg7Wg8Wg9Wh0Wh1Wh2Wh3Wh4Wh5Wh6Wh7Wh8Wh9Wi0Wi1Wi2Wi3Wi4Wi5Wi6Wi7Wi8Wi9Wj0Wj1Wj2Wj3Wj4Wj5Wj6Wj7Wj8Wj9Wk0Wk1Wk2Wk3Wk4Wk5Wk6Wk7Wk8Wk9Wl0Wl1Wl2Wl3Wl4Wl5Wl6Wl7Wl8Wl9Wm0Wm1Wm2Wm3Wm4Wm5Wm6Wm7Wm8Wm9Wn0Wn1Wn2Wn3Wn4Wn5Wn6Wn7Wn8Wn9Wo0Wo1Wo2Wo3Wo4Wo5Wo6Wo7Wo8Wo9Wp0Wp1Wp2Wp3Wp4Wp5Wp6Wp7Wp8Wp9Wq0Wq1Wq2Wq3Wq4Wq5Wq6Wq7Wq8Wq9Wr0Wr1Wr2Wr3Wr4Wr5Wr6Wr7Wr8Wr9Ws0Ws1Ws2Ws3Ws4Ws5Ws6Ws7Ws8Ws9Wt0Wt1Wt2Wt3Wt4Wt5Wt6Wt7Wt8Wt9Wu0Wu1Wu2Wu3Wu4Wu5Wu6Wu7Wu8Wu9Wv0Wv1Wv2Wv3Wv4Wv5Wv6Wv7Wv8Wv9Ww0Ww1Ww2Ww3Ww4Ww5Ww6Ww7Ww8Ww9Wx0Wx1Wx2Wx3Wx4Wx5Wx6Wx7Wx8Wx9Wy0Wy1Wy2Wy3Wy4Wy5Wy6Wy7Wy8Wy9Wz0Wz1Wz2Wz3Wz4Wz5Wz6Wz7Wz8Wz9Xa0Xa1Xa2Xa3Xa4Xa5Xa6Xa7Xa8Xa9Xb0Xb1Xb2Xb3Xb4Xb5Xb6Xb7Xb8Xb9Xc0Xc1Xc2Xc3Xc4Xc5Xc6Xc7Xc8Xc9Xd0Xd1Xd2Xd3Xd4Xd5Xd6Xd7Xd8Xd9Xe0Xe1Xe2Xe3Xe4Xe5Xe6Xe7Xe8Xe9Xf0Xf1Xf2Xf3Xf4Xf5Xf6Xf7Xf8Xf9Xg0Xg1Xg2Xg3Xg4Xg5Xg6Xg7Xg8Xg9Xh0Xh1Xh2Xh3Xh4Xh5Xh6Xh7Xh8Xh9Xi0Xi1Xi2Xi3Xi4Xi5Xi6Xi7Xi8Xi9Xj0Xj1Xj2Xj3Xj4Xj5Xj6Xj7Xj8Xj9Xk0Xk1Xk2Xk3Xk4Xk5Xk6Xk7Xk8Xk9Xl0Xl1Xl2Xl3Xl4Xl5Xl6Xl7Xl8Xl9Xm0Xm1Xm2Xm3Xm4Xm5Xm6Xm7Xm8Xm9Xn0Xn1Xn2Xn3Xn4Xn5Xn6Xn7Xn8Xn9Xo0Xo1Xo2Xo3Xo4Xo5Xo6Xo7Xo8Xo9Xp0Xp1Xp2Xp3Xp4Xp5Xp6Xp7Xp8Xp9Xq0Xq1Xq2Xq3Xq4Xq5Xq6Xq7Xq8Xq9Xr0Xr1Xr2Xr3Xr4Xr5Xr6Xr7Xr8Xr9Xs0Xs1Xs2Xs3Xs4Xs5Xs6Xs7Xs8Xs9Xt0Xt1Xt2Xt3Xt4Xt5Xt6Xt7Xt8Xt9Xu0Xu1Xu2Xu3Xu4Xu5Xu6Xu7Xu8Xu9Xv0Xv1Xv2Xv3Xv4Xv5Xv6Xv7Xv8Xv9Xw0Xw1Xw2Xw3Xw4Xw5Xw6Xw7Xw8Xw9Xx0Xx1Xx2Xx3Xx4Xx5Xx6Xx7Xx8Xx9Xy0Xy1Xy2Xy3Xy4Xy5Xy6Xy7Xy8Xy9Xz0Xz1Xz2Xz3Xz4Xz5Xz6Xz7Xz8Xz9Ya0Ya1Ya2Ya3Ya4Ya5Ya6Ya7Ya8Ya9Yb0Yb1Yb2Yb3Yb4Yb5Yb6Yb7Yb8Yb9Yc0Yc1Yc2Yc3Yc4Yc5Yc6Yc7Yc8Yc9Yd0Yd1Yd2Yd3Yd4Yd5Yd6Yd7Yd8Yd9Ye0Ye1Ye2Ye3Ye4Ye5Ye6Ye7Ye8Ye9Yf0Yf1Yf2Yf3Yf4Yf5Yf6Yf7Yf8Yf9Yg0Yg1Yg2Yg3Yg4Yg5Yg6Yg7Yg8Yg9Yh0Yh1Yh2Yh3Yh4Yh5Yh6Yh7Yh8Yh9Yi0Yi1Yi2Yi3Yi4Yi5Yi6Yi7Yi8Yi9Yj0Yj1Yj2Yj3Yj4Yj5Yj6Yj7Yj8Yj9Yk0Yk1Yk2Yk3Yk4Yk5Yk6Yk7Yk8Yk9Yl0Yl1Yl2Yl3Yl4Yl5Yl6Yl7Yl8Yl9Ym0Ym1Ym2Ym3Ym4Ym5Ym6Ym7Ym8Ym9Yn0Yn1Yn2Yn3Yn4Yn5Yn6Yn7Yn8Yn9Yo0Yo1Yo2Yo3Yo4Yo5Yo6Yo7Yo8Yo9Yp0Yp1Yp2Yp3Yp4Yp5Yp6Yp7Yp8Yp9Yq0Yq1Yq2Yq3Yq4Yq5Yq6Yq7Yq8Yq9Yr0Yr1Yr2Yr3Yr4Yr5Yr6Yr7Yr8Yr9Ys0Ys1Ys2Ys3Ys4Ys5Ys6Ys7Ys8Ys9Yt0Yt1Yt2Yt3Yt4Yt5Yt6Yt7Yt8Yt9Yu0Yu1Yu2Yu3Yu4Yu5Yu6Yu7Yu8Yu9Yv0Yv1Yv2Yv3Yv4Yv5Yv6Yv7Yv8Yv9Yw0Yw1Yw2Yw3Yw4Yw5Yw6Yw7Yw8Yw9Yx0Yx1Yx2Yx3Yx4Yx5Yx6Yx7Yx8Yx9Yy0Yy1Yy2Yy3Yy4Yy5Yy6Yy7Yy8Yy9Yz0Yz1Yz2Yz3Yz4Yz5Yz6Yz7Yz8Yz9Za0Za1Za2Za3Za4Za5Za6Za7Za8Za9Zb0Zb1Zb2Zb3Zb4Zb5Zb6Zb7Zb8Zb9Zc0Zc1Zc2Zc3Zc4Zc5Zc6Zc7Zc8Zc9Zd0Zd1Zd2Zd3Zd4Zd5Zd6Zd7Zd8Zd9Ze0Ze1Ze2Ze3Ze4Ze5Ze6Ze7Ze8Ze9Zf0Zf1Zf2Zf3Zf4Zf5Zf6Zf7Zf8Zf9Zg0Zg1Zg2Zg3Zg4Zg5Zg6Zg7Zg8Zg9Zh0Zh1Zh2Zh3Zh4Zh5Zh6Zh7Zh8Zh9Zi0Zi1Zi2Zi3Zi4Zi5Zi6Zi7Zi8Zi9Zj0Zj1Zj2Zj3Zj4Zj5Zj6Zj7Zj8Zj9Zk0Zk1Zk2Zk3Zk4Zk5Zk6Zk7Zk8Zk9Zl0Zl1Zl2Zl3Zl4Zl5Zl6Zl7Zl8Zl9Zm0Zm1Zm2Zm3Zm4Zm5Zm6Zm7Zm8Zm9Zn0Zn1Zn2Zn3Zn4Zn5Zn6Zn7Zn8Zn9Zo0Zo1Zo2Zo3Zo4Zo5Zo6Zo7Zo8Zo9Zp0Zp1Zp2Zp3Zp4Zp5Zp6Zp7Zp8Zp9Zq0Zq1Zq2Zq3Zq4Zq5Zq";

my $maki =
"\x46\x47" .                              # Magic
"\x03\x04" .                              # Version
"\x17\x00\x00\x00" .                      # ???
"\x01\x00\x00\x00" .                      # Types count
"\x71\x49\x65\x51\x87\x0D\x51\x4A" .      # Types
"\x91\xE3\xA6\xB5\x32\x35\xF3\xE7" .

"\x01\x00\x00\x00".                       # Function count
"\x01\x01" .                              # Function 1
"\x00\x00" .                              # Dummy

$length .                                 # Length
$function_name;

print $maki;
