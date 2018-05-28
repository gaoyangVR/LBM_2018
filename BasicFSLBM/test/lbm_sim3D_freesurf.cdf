(* Content-type: application/vnd.wolfram.cdf.text *)

(*** Wolfram CDF File ***)
(* http://www.wolfram.com/cdf *)

(* CreatedBy='Mathematica 10.0' *)

(*************************************************************************)
(*                                                                       *)
(*  The Mathematica License under which this file was created prohibits  *)
(*  restricting third parties in receipt of this file from republishing  *)
(*  or redistributing it by any means, including but not limited to      *)
(*  rights management or terms of use, without the express consent of    *)
(*  Wolfram Research, Inc. For additional information concerning CDF     *)
(*  licensing and redistribution see:                                    *)
(*                                                                       *)
(*        www.wolfram.com/cdf/adopting-cdf/licensing-options.html        *)
(*                                                                       *)
(*************************************************************************)

(*CacheID: 234*)
(* Internal cache information:
NotebookFileLineBreakTest
NotebookFileLineBreakTest
NotebookDataPosition[      1064,         20]
NotebookDataLength[    531460,      10153]
NotebookOptionsPosition[    527703,      10004]
NotebookOutlinePosition[    528253,      10027]
CellTagsIndexPosition[    528165,      10022]
WindowFrame->Normal*)

(* Beginning of Notebook Content *)
Notebook[{

Cell[CellGroupData[{
Cell["Lattice Boltzmann Method (LBM)", "Title"],

Cell[TextData[{
 "Implementation of the lattice Boltzmann method (LBM) using the D2Q9 and \
D3Q19 models\n\nCopyright (c) 2014, Christian B. Mendl\nAll rights reserved.\n\
",
 ButtonBox["http://christian.mendl.net",
  BaseStyle->"Hyperlink",
  ButtonData->{
    URL["http://christian.mendl.net"], None},
  ButtonNote->"http://christian.mendl.net"],
 "\n\nThis program is free software; you can redistribute it and/or\nmodify \
it under the terms of the Simplified BSD License\n",
 ButtonBox["http://www.opensource.org/licenses/bsd-license.php",
  BaseStyle->"Hyperlink",
  ButtonData->{
    URL["http://www.opensource.org/licenses/bsd-license.php"], None},
  ButtonNote->"http://www.opensource.org/licenses/bsd-license.php"],
 "\n\nReference:\n\tNils Th\[UDoubleDot]rey, Physically based animation of \
free surface flows with the lattice Boltzmann method,\n\tPhD thesis, \
University of Erlangen-Nuremberg (2007)"
}], "Text"],

Cell[BoxData[
 RowBox[{
  RowBox[{"SetDirectory", "[", 
   RowBox[{"NotebookDirectory", "[", "]"}], "]"}], ";"}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"lbmLink", "=", 
   RowBox[{"Install", "[", 
    RowBox[{"\"\<../mlink/\>\"", "<>", "$SystemID", "<>", "\"\</lbmWS\>\""}], 
    "]"}]}], ";"}]], "Input"],

Cell[BoxData[
 RowBox[{"(*", " ", 
  RowBox[{
   RowBox[{
   "to", " ", "use", " ", "the", " ", "traditional", " ", "MathLink", " ", 
    "interface"}], ",", " ", 
   RowBox[{
    RowBox[{
     RowBox[{"call", "\[IndentingNewLine]", "lbmLink"}], "=", 
     RowBox[{"Install", "[", 
      RowBox[{
      "\"\<../mlink/\>\"", "<>", "$SystemID", "<>", "\"\</lbmML\>\""}], 
      "]"}]}], ";"}]}], "*)"}]], "Input"],

Cell[CellGroupData[{

Cell["Common functions", "Section"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   SubscriptBox["vel", "3"], "=", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"0", ",", "0", ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "1"}], ",", "0", ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "0", ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", 
       RowBox[{"-", "1"}], ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", "1", ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", "0", ",", 
       RowBox[{"-", "1"}]}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", "0", ",", "1"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "1"}], ",", 
       RowBox[{"-", "1"}], ",", "0"}], "}"}], ",", "\n", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "1"}], ",", "1", ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", 
       RowBox[{"-", "1"}], ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "1", ",", "0"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", 
       RowBox[{"-", "1"}], ",", 
       RowBox[{"-", "1"}]}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", 
       RowBox[{"-", "1"}], ",", "1"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", "1", ",", 
       RowBox[{"-", "1"}]}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"0", ",", "1", ",", "1"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "1"}], ",", "0", ",", 
       RowBox[{"-", "1"}]}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "1"}], ",", "0", ",", "1"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "0", ",", 
       RowBox[{"-", "1"}]}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "0", ",", "1"}], "}"}]}], "}"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{"Length", "[", "%", "]"}]}], "Input"],

Cell[BoxData["19"], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"Graphics3D", "[", 
  RowBox[{
   RowBox[{"Arrow", "[", 
    RowBox[{
     RowBox[{
      RowBox[{"{", 
       RowBox[{
        RowBox[{"{", 
         RowBox[{"0", ",", "0", ",", "0"}], "}"}], ",", "#"}], "}"}], "&"}], "/@", 
     SubscriptBox["vel", "3"]}], "]"}], ",", 
   RowBox[{"ImageSize", "\[Rule]", "Small"}]}], "]"}]], "Input"],

Cell[BoxData[
 Graphics3DBox[Arrow3DBox[CompressedData["
1:eJxTTMoPymNmYGAQBmImIAaxiQH/gQCbOOMA6fkPBdj0YzOfEYfZjHjcQ0s7
YOrQ1cP46OoZ0eTRxUE0ALv6PaE=
   "]],
  ImageSize->Small]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   SubscriptBox["weights", "3"], "=", 
   RowBox[{"Join", "[", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"1", "/", "3"}], "}"}], ",", 
     RowBox[{"ConstantArray", "[", 
      RowBox[{
       RowBox[{"1", "/", "18"}], ",", "6"}], "]"}], ",", 
     RowBox[{"ConstantArray", "[", 
      RowBox[{
       RowBox[{"1", "/", "36"}], ",", "12"}], "]"}]}], "]"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{"Length", "[", "%", "]"}]}], "Input"],

Cell[BoxData["19"], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", "check", " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{"Total", "[", 
    SubscriptBox["vel", "3"], "]"}], "\[IndentingNewLine]", 
   RowBox[{"Total", "[", 
    SubscriptBox["weights", "3"], "]"}]}]}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"0", ",", "0", ",", "0"}], "}"}]], "Output"],

Cell[BoxData["1"], "Output"]
}, Open  ]],

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"LBM", " ", "equilibrium", " ", "distribution"}], " ", "*)"}], 
  "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{"LBMeq", "[", 
    RowBox[{"\[Rho]_", ",", "u_"}], "]"}], ":=", 
   RowBox[{"Table", "[", 
    RowBox[{
     RowBox[{"\[Rho]", " ", 
      RowBox[{
       SubscriptBox["weights", "3"], "\[LeftDoubleBracket]", "i", 
       "\[RightDoubleBracket]"}], 
      RowBox[{"(", 
       RowBox[{"1", "+", 
        RowBox[{"3", 
         RowBox[{
          RowBox[{
           SubscriptBox["vel", "3"], "\[LeftDoubleBracket]", "i", 
           "\[RightDoubleBracket]"}], ".", "u"}]}], "-", 
        RowBox[{
         FractionBox["3", "2"], 
         RowBox[{"u", ".", "u"}]}], "+", 
        RowBox[{
         FractionBox["9", "2"], 
         SuperscriptBox[
          RowBox[{"(", 
           RowBox[{
            RowBox[{
             SubscriptBox["vel", "3"], "\[LeftDoubleBracket]", "i", 
             "\[RightDoubleBracket]"}], ".", "u"}], ")"}], "2"]}]}], ")"}]}], 
     ",", 
     RowBox[{"{", 
      RowBox[{"i", ",", 
       RowBox[{"Length", "[", 
        SubscriptBox["weights", "3"], "]"}]}], "}"}]}], "]"}]}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"Density", "[", "f_", "]"}], ":=", 
  RowBox[{"Total", "[", "f", "]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"Velocity", "[", "f_", "]"}], ":=", 
  RowBox[{"Module", "[", 
   RowBox[{
    RowBox[{"{", 
     RowBox[{"n", "=", 
      RowBox[{"Density", "[", "f", "]"}]}], "}"}], ",", 
    RowBox[{"If", "[", 
     RowBox[{
      RowBox[{"n", ">", "0"}], ",", 
      RowBox[{
       FractionBox["1", "n"], 
       RowBox[{"Sum", "[", 
        RowBox[{
         RowBox[{
          RowBox[{"f", "\[LeftDoubleBracket]", "i", "\[RightDoubleBracket]"}], 
          RowBox[{
           SubscriptBox["vel", "3"], "\[LeftDoubleBracket]", "i", 
           "\[RightDoubleBracket]"}]}], ",", 
         RowBox[{"{", 
          RowBox[{"i", ",", 
           RowBox[{"Length", "[", "f", "]"}]}], "}"}]}], "]"}]}], ",", 
      RowBox[{"{", 
       RowBox[{"0", ",", "0", ",", "0"}], "}"}]}], "]"}]}], "]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"InternalEnergy", "[", "f_", "]"}], ":=", 
  RowBox[{"Module", "[", 
   RowBox[{
    RowBox[{"{", 
     RowBox[{
      RowBox[{"n", "=", 
       RowBox[{"Density", "[", "f", "]"}]}], ",", 
      RowBox[{"u", "=", 
       RowBox[{"Velocity", "[", "f", "]"}]}]}], "}"}], ",", 
    RowBox[{"If", "[", 
     RowBox[{
      RowBox[{"n", ">", "0"}], ",", 
      RowBox[{
       FractionBox["1", "n"], 
       RowBox[{"Sum", "[", 
        RowBox[{
         RowBox[{
          FractionBox["1", "2"], 
          SuperscriptBox[
           RowBox[{"Norm", "[", 
            RowBox[{
             RowBox[{
              SubscriptBox["vel", "3"], "\[LeftDoubleBracket]", "i", 
              "\[RightDoubleBracket]"}], "-", "u"}], "]"}], "2"], 
          RowBox[{
          "f", "\[LeftDoubleBracket]", "i", "\[RightDoubleBracket]"}]}], ",", 
         RowBox[{"{", 
          RowBox[{"i", ",", 
           RowBox[{"Length", "[", "f", "]"}]}], "}"}]}], "]"}]}], ",", "0"}], 
     "]"}]}], "]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"TotalEnergy", "[", "f_", "]"}], ":=", 
  RowBox[{"Module", "[", 
   RowBox[{
    RowBox[{"{", 
     RowBox[{"n", "=", 
      RowBox[{"Density", "[", "f", "]"}]}], "}"}], ",", 
    RowBox[{"If", "[", 
     RowBox[{
      RowBox[{"n", ">", "0"}], ",", 
      RowBox[{
       FractionBox["1", "n"], 
       RowBox[{"Sum", "[", 
        RowBox[{
         RowBox[{
          FractionBox["1", "2"], 
          SuperscriptBox[
           RowBox[{"Norm", "[", 
            RowBox[{
             SubscriptBox["vel", "3"], "\[LeftDoubleBracket]", "i", 
             "\[RightDoubleBracket]"}], "]"}], "2"], 
          RowBox[{
          "f", "\[LeftDoubleBracket]", "i", "\[RightDoubleBracket]"}]}], ",", 
         RowBox[{"{", 
          RowBox[{"i", ",", 
           RowBox[{"Length", "[", "f", "]"}]}], "}"}]}], "]"}]}], ",", "0"}], 
     "]"}]}], "]"}]}]], "Input"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"check", ":", " ", "density"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{"FullSimplify", "[", 
   RowBox[{"Density", "[", 
    RowBox[{"LBMeq", "[", 
     RowBox[{"\[Rho]", ",", 
      RowBox[{"{", 
       RowBox[{
        SubscriptBox["u", "1"], ",", 
        SubscriptBox["u", "2"], ",", 
        SubscriptBox["u", "3"]}], "}"}]}], "]"}], "]"}], "]"}]}]], "Input"],

Cell[BoxData["\[Rho]"], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"check", ":", " ", 
    RowBox[{"average", " ", "velocity"}]}], " ", "*)"}], 
  "\[IndentingNewLine]", 
  RowBox[{"FullSimplify", "[", 
   RowBox[{
    RowBox[{"Velocity", "[", 
     RowBox[{"LBMeq", "[", 
      RowBox[{"\[Rho]", ",", 
       RowBox[{"{", 
        RowBox[{
         SubscriptBox["u", "1"], ",", 
         SubscriptBox["u", "2"], ",", 
         SubscriptBox["u", "3"]}], "}"}]}], "]"}], "]"}], ",", 
    RowBox[{"Assumptions", "\[RuleDelayed]", 
     RowBox[{"{", 
      RowBox[{"\[Rho]", ">", "0"}], "}"}]}]}], "]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{
   SubscriptBox["u", "1"], ",", 
   SubscriptBox["u", "2"], ",", 
   SubscriptBox["u", "3"]}], "}"}]], "Output"]
}, Open  ]],

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"cell", " ", "types"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{
     SubscriptBox["ct", "obstacle"], "=", 
     SuperscriptBox["2", "0"]}], ";"}], "\[IndentingNewLine]", 
   RowBox[{
    RowBox[{
     SubscriptBox["ct", "fluid"], "=", 
     SuperscriptBox["2", "1"]}], ";"}], "\[IndentingNewLine]", 
   RowBox[{
    RowBox[{
     SubscriptBox["ct", "interface"], "=", 
     SuperscriptBox["2", "2"]}], ";"}], "\[IndentingNewLine]", 
   RowBox[{
    RowBox[{
     SubscriptBox["ct", "empty"], "=", 
     SuperscriptBox["2", "3"]}], ";"}]}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"VisualizeTypeField", "[", 
   RowBox[{"typefield_", ",", 
    RowBox[{"plotlabel_:", "None"}]}], "]"}], ":=", 
  RowBox[{"Graphics3D", "[", 
   RowBox[{
    RowBox[{"Raster3D", "[", 
     RowBox[{"Map", "[", 
      RowBox[{
       RowBox[{
        RowBox[{"#", "/.", 
         RowBox[{"{", 
          RowBox[{
           RowBox[{
            SubscriptBox["ct", "obstacle"], "\[Rule]", 
            RowBox[{"{", 
             RowBox[{"1", ",", "0", ",", "0", ",", 
              RowBox[{"(*", "transparent", "*)"}], "0"}], "}"}]}], ",", 
           RowBox[{
            SubscriptBox["ct", "fluid"], "\[Rule]", 
            RowBox[{"{", 
             RowBox[{"0", ",", "0", ",", "1", ",", "1"}], "}"}]}], ",", 
           RowBox[{
            SubscriptBox["ct", "interface"], "\[Rule]", 
            RowBox[{"{", 
             RowBox[{
              RowBox[{"1", "/", "2"}], ",", 
              RowBox[{"1", "/", "2"}], ",", "1", ",", 
              RowBox[{"1", "/", "2"}]}], "}"}]}], ",", 
           RowBox[{
            SubscriptBox["ct", "empty"], "\[Rule]", 
            RowBox[{"{", 
             RowBox[{"1", ",", "1", ",", "1", ",", 
              RowBox[{"(*", "transparent", "*)"}], "0"}], "}"}]}]}], "}"}]}], 
        "&"}], ",", 
       RowBox[{"Transpose", "[", 
        RowBox[{"typefield", ",", 
         RowBox[{"{", 
          RowBox[{"3", ",", "2", ",", "1"}], "}"}]}], "]"}], ",", 
       RowBox[{"{", "3", "}"}]}], "]"}], "]"}], ",", 
    RowBox[{"Axes", "\[Rule]", "True"}], ",", 
    RowBox[{"AxesLabel", "\[Rule]", 
     RowBox[{"{", 
      RowBox[{"\"\<x\>\"", ",", "\"\<y\>\"", ",", "\"\<z\>\""}], "}"}]}], ",", 
    RowBox[{"PlotLabel", "\[Rule]", "plotlabel"}], ",", 
    RowBox[{"ViewPoint", "\[Rule]", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "2.4"}], ",", "1.1", ",", "1.0"}], "}"}]}]}], 
   "]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"VisualizeVelocityField", "[", 
   RowBox[{"vel_", ",", 
    RowBox[{"plotlabel_:", "None"}]}], "]"}], ":=", 
  RowBox[{"ListVectorPlot3D", "[", 
   RowBox[{"vel", ",", 
    RowBox[{"AxesLabel", "\[Rule]", 
     RowBox[{"{", 
      RowBox[{"\"\<x\>\"", ",", "\"\<y\>\"", ",", "\"\<z\>\""}], "}"}]}], ",", 
    RowBox[{"VectorScale", "\[Rule]", "Small"}], ",", 
    RowBox[{"PlotLabel", "\[Rule]", "plotlabel"}], ",", 
    RowBox[{"VectorStyle", "\[Rule]", "Blue"}], ",", 
    RowBox[{"ViewPoint", "\[Rule]", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "2.4"}], ",", "1.1", ",", "1.0"}], "}"}]}]}], 
   "]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"Visualize3DTable", "[", "tab_", "]"}], ":=", 
  RowBox[{"Image3D", "[", 
   RowBox[{
    RowBox[{"Reverse", "[", 
     RowBox[{
      RowBox[{"Transpose", "[", 
       RowBox[{"tab", ",", 
        RowBox[{"{", 
         RowBox[{"3", ",", "2", ",", "1"}], "}"}]}], "]"}], ",", 
      RowBox[{"{", 
       RowBox[{"1", ",", "2"}], "}"}]}], "]"}], ",", 
    RowBox[{"Boxed", "\[Rule]", "True"}], ",", 
    RowBox[{"Axes", "\[Rule]", "True"}], ",", 
    RowBox[{"AxesLabel", "\[Rule]", 
     RowBox[{"{", 
      RowBox[{"\"\<x\>\"", ",", "\"\<y\>\"", ",", "\"\<z\>\""}], "}"}]}], ",", 
    RowBox[{"ViewPoint", "\[Rule]", 
     RowBox[{"{", 
      RowBox[{
       RowBox[{"-", "2.4"}], ",", "1.1", ",", "1.0"}], "}"}]}]}], 
   "]"}]}]], "Input"]
}, Open  ]],

Cell[CellGroupData[{

Cell["Define initial flow field", "Section"],

Cell[BoxData[{
 RowBox[{
  RowBox[{
   SubscriptBox["dim", "1"], "=", "8"}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   SubscriptBox["dim", "2"], "=", "16"}], ";"}], "\[IndentingNewLine]", 
 RowBox[{
  RowBox[{
   SubscriptBox["dim", "3"], "=", "32"}], ";"}]}], "Input"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   SubscriptBox["type", "0"], "=", 
   RowBox[{"Block", "[", 
    RowBox[{
     RowBox[{"{", "t0", "}"}], ",", 
     RowBox[{
      RowBox[{"t0", "=", 
       RowBox[{"Table", "[", 
        RowBox[{
         RowBox[{"If", "[", 
          RowBox[{
           RowBox[{
            RowBox[{"i", "\[Equal]", "1"}], "\[Or]", 
            RowBox[{"i", "\[Equal]", 
             SubscriptBox["dim", "1"]}], "\[Or]", 
            RowBox[{"j", "\[Equal]", "1"}], "\[Or]", 
            RowBox[{"j", "\[Equal]", 
             SubscriptBox["dim", "2"]}], "\[Or]", 
            RowBox[{"k", "\[Equal]", "1"}], "\[Or]", 
            RowBox[{"k", "\[Equal]", 
             SubscriptBox["dim", "3"]}]}], ",", 
           SubscriptBox["ct", "obstacle"], ",", 
           RowBox[{"If", "[", 
            RowBox[{
             RowBox[{
              RowBox[{
               RowBox[{"Norm", "[", 
                RowBox[{"{", 
                 RowBox[{
                  RowBox[{"i", "-", 
                   RowBox[{
                    SubscriptBox["dim", "1"], "/", "2"}], "-", "1"}], ",", 
                  RowBox[{"j", "-", 
                   RowBox[{
                    SubscriptBox["dim", "2"], "/", "2"}], "-", "1"}], ",", 
                  RowBox[{"k", "-", 
                   RowBox[{"3", 
                    RowBox[{
                    SubscriptBox["dim", "3"], "/", "4"}]}], "-", "1"}]}], 
                 "}"}], "]"}], "\[LessEqual]", 
               RowBox[{
                SubscriptBox["dim", "1"], "/", "4"}]}], "\[Or]", 
              RowBox[{"k", "\[LessEqual]", "3"}]}], ",", 
             SubscriptBox["ct", "fluid"], ",", 
             SubscriptBox["ct", "interface"]}], "]"}]}], "]"}], ",", 
         RowBox[{"{", 
          RowBox[{"i", ",", 
           SubscriptBox["dim", "1"]}], "}"}], ",", 
         RowBox[{"{", 
          RowBox[{"j", ",", 
           SubscriptBox["dim", "2"]}], "}"}], ",", 
         RowBox[{"{", 
          RowBox[{"k", ",", 
           SubscriptBox["dim", "3"]}], "}"}]}], "]"}]}], ";", 
      RowBox[{"Table", "[", 
       RowBox[{
        RowBox[{"If", "[", 
         RowBox[{
          RowBox[{
           RowBox[{"Norm", "[", 
            RowBox[{"Flatten", "[", 
             RowBox[{
              RowBox[{"(", 
               RowBox[{
                RowBox[{"t0", "\[LeftDoubleBracket]", 
                 RowBox[{
                  RowBox[{
                   RowBox[{"i", "-", "1"}], ";;", 
                   RowBox[{"i", "+", "1"}]}], ",", 
                  RowBox[{
                   RowBox[{"j", "-", "1"}], ";;", 
                   RowBox[{"j", "+", "1"}]}], ",", 
                  RowBox[{
                   RowBox[{"k", "-", "1"}], ";;", 
                   RowBox[{"k", "+", "1"}]}]}], "\[RightDoubleBracket]"}], "/.", 
                RowBox[{"{", 
                 RowBox[{
                  RowBox[{
                   SubscriptBox["ct", "obstacle"], "\[Rule]", 
                   SubscriptBox["ct", "interface"]}], ",", 
                  RowBox[{
                   SubscriptBox["ct", "empty"], "\[Rule]", 
                   SubscriptBox["ct", "interface"]}]}], "}"}]}], ")"}], "-", 
              SubscriptBox["ct", "interface"]}], "]"}], "]"}], "\[Equal]", 
           "0"}], ",", 
          RowBox[{
           RowBox[{"t0", "\[LeftDoubleBracket]", 
            RowBox[{"i", ",", "j", ",", "k"}], "\[RightDoubleBracket]"}], "=", 
           SubscriptBox["ct", "empty"]}]}], "]"}], ",", 
        RowBox[{"{", 
         RowBox[{"i", ",", "2", ",", 
          RowBox[{
           SubscriptBox["dim", "1"], "-", "1"}]}], "}"}], ",", 
        RowBox[{"{", 
         RowBox[{"j", ",", "2", ",", 
          RowBox[{
           SubscriptBox["dim", "2"], "-", "1"}]}], "}"}], ",", 
        RowBox[{"{", 
         RowBox[{"k", ",", "2", ",", 
          RowBox[{
           SubscriptBox["dim", "3"], "-", "1"}]}], "}"}]}], "]"}], ";", 
      "t0"}]}], "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{"Dimensions", "[", 
  SubscriptBox["type", "0"], "]"}]}], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"8", ",", "16", ",", "32"}], "}"}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", "example", " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   SubscriptBox["type", "0"], "\[LeftDoubleBracket]", 
   RowBox[{
    RowBox[{"1", ";;", "2"}], ",", 
    RowBox[{"1", ";;", "3"}], ",", 
    RowBox[{"1", ";;", "2"}]}], "\[RightDoubleBracket]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"1", ",", "1"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "1"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "1"}], "}"}]}], "}"}], ",", 
   RowBox[{"{", 
    RowBox[{
     RowBox[{"{", 
      RowBox[{"1", ",", "1"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "2"}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"1", ",", "2"}], "}"}]}], "}"}]}], "}"}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"VisualizeTypeField", "[", 
  SubscriptBox["type", "0"], "]"}]], "Input"],

Cell[BoxData[
 Graphics3DBox[Raster3DBox[{CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYsZRPKIxAC1xAkY=
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkYiMJBkxIZH5YeGPD4MAIB1Apo=
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkYiMJBkxIZH5YeGPD4MAIB1Apo=
    "], {{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
     0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 
     0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], {{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1,
      1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, {{{1, 0, 0, 0}, {1, 0, 0,
      0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
     0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, {{{1, 0, 0,
      0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1,
      1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, {{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0,
      0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1,
       1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {1, 0, 0, 
      0}}, {{1, 0, 0, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1,
      1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, {{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0,
      0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {0, 0,
       1, 1}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1,
      1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, {{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0,
      0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {0, 0, 1, 1}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
      0, 0, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 0, 0, 0}}, {{1, 
     0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, {{{1, 0, 0,
      0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {
      Rational[1, 2], Rational[1, 2], 1, Rational[1, 2]}, {1, 1, 1, 0}, {1, 0,
       0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1,
      1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 
     1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 0, 0, 0}}, {{1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 
     0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 
     0}}, {{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 
     0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}}, CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYkZiMCMjdjwqPzTk8WAA1cUC7g==
    "], CompressedData["
1:eJxTTMoPSmJmYGAQAGIOIGYBYsZRPKIxAC1xAkY=
    "]}],
  Axes->True,
  AxesLabel->{
    FormBox["\"x\"", TraditionalForm], 
    FormBox["\"y\"", TraditionalForm], 
    FormBox["\"z\"", TraditionalForm]},
  PlotLabel->None,
  ViewPoint->{-2.4, 1.1, 1.}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   SubscriptBox["f", "0"], "=", 
   RowBox[{"N", "[", 
    RowBox[{"Table", "[", 
     RowBox[{
      RowBox[{"If", "[", 
       RowBox[{
        RowBox[{
         RowBox[{
          RowBox[{
           SubscriptBox["type", "0"], "\[LeftDoubleBracket]", 
           RowBox[{"i", ",", "j", ",", "k"}], "\[RightDoubleBracket]"}], 
          "\[Equal]", 
          SubscriptBox["ct", "fluid"]}], "\[Or]", 
         RowBox[{
          RowBox[{
           SubscriptBox["type", "0"], "\[LeftDoubleBracket]", 
           RowBox[{"i", ",", "j", ",", "k"}], "\[RightDoubleBracket]"}], 
          "\[Equal]", 
          SubscriptBox["ct", "interface"]}]}], ",", 
        RowBox[{"LBMeq", "[", 
         RowBox[{"1", ",", 
          RowBox[{"{", 
           RowBox[{"0", ",", 
            FractionBox["1", "5"], ",", "0"}], "}"}]}], "]"}], ",", 
        RowBox[{"LBMeq", "[", 
         RowBox[{"0", ",", 
          RowBox[{"{", 
           RowBox[{"0", ",", "0", ",", "0"}], "}"}]}], "]"}]}], "]"}], ",", 
      RowBox[{"{", 
       RowBox[{"i", ",", 
        SubscriptBox["dim", "1"]}], "}"}], ",", 
      RowBox[{"{", 
       RowBox[{"j", ",", 
        SubscriptBox["dim", "2"]}], "}"}], ",", 
      RowBox[{"{", 
       RowBox[{"k", ",", 
        SubscriptBox["dim", "3"]}], "}"}]}], "]"}], "]"}]}], 
  ";"}], "\[IndentingNewLine]", 
 RowBox[{"Dimensions", "[", 
  SubscriptBox["f", "0"], "]"}]}], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"8", ",", "16", ",", "32", ",", "19"}], "}"}]], "Output"]
}, Open  ]],

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"save", " ", "to", " ", "disk"}], " ", "*)"}], 
  "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{"Export", "[", 
     RowBox[{
      RowBox[{
       RowBox[{"FileBaseName", "[", 
        RowBox[{"NotebookFileName", "[", "]"}], "]"}], "<>", 
       "\"\<_t0.dat\>\""}], ",", 
      RowBox[{"Flatten", "[", 
       SubscriptBox["type", "0"], "]"}], ",", "\"\<Integer32\>\""}], "]"}], 
    ";"}], "\[IndentingNewLine]", 
   RowBox[{
    RowBox[{"Export", "[", 
     RowBox[{
      RowBox[{
       RowBox[{"FileBaseName", "[", 
        RowBox[{"NotebookFileName", "[", "]"}], "]"}], "<>", 
       "\"\<_f0.dat\>\""}], ",", 
      RowBox[{"Flatten", "[", 
       SubscriptBox["f", "0"], "]"}], ",", "\"\<Real32\>\""}], "]"}], 
    ";"}]}]}]], "Input"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"initial", " ", "velocity", " ", "u"}], " ", "*)"}], 
  "\[IndentingNewLine]", 
  RowBox[{"VisualizeVelocityField", "[", 
   RowBox[{"Table", "[", 
    RowBox[{
     RowBox[{"Velocity", "[", 
      RowBox[{
       SubscriptBox["f", "0"], "\[LeftDoubleBracket]", 
       RowBox[{"i", ",", "j", ",", "k"}], "\[RightDoubleBracket]"}], "]"}], 
     ",", 
     RowBox[{"{", 
      RowBox[{"i", ",", 
       SubscriptBox["dim", "1"]}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"j", ",", 
       SubscriptBox["dim", "2"]}], "}"}], ",", 
     RowBox[{"{", 
      RowBox[{"k", ",", 
       SubscriptBox["dim", "3"]}], "}"}]}], "]"}], "]"}]}]], "Input"],

Cell[BoxData[
 Graphics3DBox[{{}, 
   {RGBColor[0, 0, 1], 
    {Arrowheads[{{0.0026867163980399283`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 3.204037396389047, 5.428571428571429}, {2., 
      3.081676889325239, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 5.34689453924619, 5.428571428571429}, {2., 
      5.224534032182381, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 7.489751682103333, 5.428571428571429}, {2., 
      7.367391175039525, 5.428571428571429}}]}, 
    {Arrowheads[{{0.002271567800092117, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 7.376844683602762, 23.142857142857146`}, {2., 
      7.480298173540095, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0010325308182236575`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 7.452083585375368, 27.571428571428577`}, {2., 
      7.405059271767491, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 9.632608824960474, 5.428571428571429}, {2., 
      9.510248317896666, 5.428571428571429}}]}, 
    {Arrowheads[{{0.005718084462300858, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 9.441219868576411, 23.142857142857146`}, {2., 
      9.701637274280731, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0025991293010457772`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 9.630614345452278, 27.571428571428577`}, {2., 
      9.512242797404863, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 11.775465967817617`, 5.428571428571429}, {2., 
      11.653105460753808`, 5.428571428571429}}]}, 
    {Arrowheads[{{0.00023498977242333255`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 11.71963675686868, 23.142857142857146`}, {2., 
      11.708934671702748`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.00010681353291970015`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 11.711853422202548`, 27.571428571428577`}, {2., 
      11.71671800636888, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{2., 13.91832311067476, 5.428571428571429}, {2., 
      13.79596260361095, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399283`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 3.204037396389047, 5.428571428571429}, {3., 
      3.081676889325239, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 5.34689453924619, 5.428571428571429}, {3., 
      5.224534032182381, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0019012808859704754`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 5.329009084794642, 23.142857142857146`}, {3., 
      5.2424194866339295`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0007743981136677382, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 5.30334840331724, 27.571428571428577`}, {3., 
      5.268080168111332, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 7.489751682103333, 5.428571428571429}, {3., 
      7.367391175039525, 5.428571428571429}}]}, 
    {Arrowheads[{{0.046264501558614994`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 6.3750646509494295`, 23.142857142857146`}, {3., 
      8.482078206193428, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.018843687432581677`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 6.999474566899553, 27.571428571428577`}, {3., 
      7.857668290243304, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 9.632608824960474, 5.428571428571429}, {3., 
      9.510248317896666, 5.428571428571429}}]}, 
    {Arrowheads[{{0.043475956259191616`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 8.581420832457761, 23.142857142857146`}, {3., 
      10.561436310399381`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.01770790353253569, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 9.168195082241027, 27.571428571428577`}, {3., 
      9.974662060616115, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 11.775465967817617`, 5.428571428571429}, {3., 
      11.653105460753808`, 5.428571428571429}}]}, 
    {Arrowheads[{{0.011787941493017002`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 11.445857959987505`, 23.142857142857146`}, {3., 
      11.982713468583922`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.004801268304739997, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 11.604954185147399`, 27.571428571428577`}, {3., 
      11.823617243424026`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{3., 13.91832311067476, 5.428571428571429}, {3., 
      13.79596260361095, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399283`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 3.204037396389047, 5.428571428571429}, {4., 
      3.081676889325239, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 5.34689453924619, 5.428571428571429}, {4., 
      5.224534032182381, 5.428571428571429}}]}, 
    {Arrowheads[{{0.00020935452452255536`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 5.29048157819729, 23.142857142857146`}, {4., 
      5.2809469932312805`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0015434555506894936`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 5.320860906316035, 27.571428571428577`}, {4., 
      5.250567665112537, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 7.489751682103333, 5.428571428571429}, {4., 
      7.367391175039525, 5.428571428571429}}]}, 
    {Arrowheads[{{0.04524870486054872, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 6.39819574866034, 23.142857142857146`}, {4., 
      8.45894710848252, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.03035818628132433, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 6.737273480334248, 27.571428571428577`}, {4., 
      8.119869376808609, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 9.632608824960474, 5.428571428571429}, {4., 
      9.510248317896666, 5.428571428571429}}]}, 
    {Arrowheads[{{0.04178652221351181, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 8.619891585573171, 23.142857142857146`}, {4., 
      10.52296555728397, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.04629832584403955, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 8.517151567980237, 27.571428571428577`}, {4., 
      10.625705574876907`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 11.775465967817617`, 5.428571428571429}, {4., 
      11.653105460753808`, 5.428571428571429}}]}, 
    {Arrowheads[{{0.01608718619303459, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 11.347958203640069`, 23.142857142857146`}, {4., 
      12.080613224931358`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0017036758500690048`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 11.675490655559216`, 27.571428571428577`}, {4., 
      11.753080773012211`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 13.91832311067476, 5.428571428571429}, {4., 
      13.79596260361095, 5.428571428571429}}]}, 
    {Arrowheads[{{0.00012532787862576175`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 13.859996746520437`, 23.142857142857146`}, {4., 
      13.854288967765275`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.00005696721755717861, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{4., 13.8558456346985, 27.571428571428577`}, {4., 
      13.858440079587211`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399283`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 3.204037396389047, 5.428571428571429}, {5., 
      3.081676889325239, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 5.34689453924619, 5.428571428571429}, {5., 
      5.224534032182381, 5.428571428571429}}]}, 
    {Arrowheads[{{0.00020935452452255536`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 5.29048157819729, 23.142857142857146`}, {5., 
      5.2809469932312805`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0015434555506894936`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 5.320860906316035, 27.571428571428577`}, {5., 
      5.250567665112537, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 7.489751682103333, 5.428571428571429}, {5., 
      7.367391175039525, 5.428571428571429}}]}, 
    {Arrowheads[{{0.04524870486054872, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 6.39819574866034, 23.142857142857146`}, {5., 
      8.45894710848252, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.03035818628132433, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 6.737273480334248, 27.571428571428577`}, {5., 
      8.119869376808609, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 9.632608824960474, 5.428571428571429}, {5., 
      9.510248317896666, 5.428571428571429}}]}, 
    {Arrowheads[{{0.04178652221351181, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 8.619891585573171, 23.142857142857146`}, {5., 
      10.52296555728397, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.04629832584403955, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 8.517151567980237, 27.571428571428577`}, {5., 
      10.625705574876907`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 11.775465967817617`, 5.428571428571429}, {5., 
      11.653105460753808`, 5.428571428571429}}]}, 
    {Arrowheads[{{0.01608718619303459, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 11.347958203640069`, 23.142857142857146`}, {5., 
      12.080613224931358`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0017036758500690048`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 11.675490655559216`, 27.571428571428577`}, {5., 
      11.753080773012211`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 13.91832311067476, 5.428571428571429}, {5., 
      13.79596260361095, 5.428571428571429}}]}, 
    {Arrowheads[{{0.00012532787862576175`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 13.859996746520437`, 23.142857142857146`}, {5., 
      13.854288967765275`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.00005696721755717861, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{5., 13.8558456346985, 27.571428571428577`}, {5., 
      13.858440079587211`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399283`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 3.204037396389047, 5.428571428571429}, {6., 
      3.081676889325239, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 5.34689453924619, 5.428571428571429}, {6., 
      5.224534032182381, 5.428571428571429}}]}, 
    {Arrowheads[{{0.00020935452452255536`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 5.29048157819729, 23.142857142857146`}, {6., 
      5.2809469932312805`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0015434555506894936`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 5.320860906316035, 27.571428571428577`}, {6., 
      5.250567665112537, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 7.489751682103333, 5.428571428571429}, {6., 
      7.367391175039525, 5.428571428571429}}]}, 
    {Arrowheads[{{0.04524870486054872, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 6.39819574866034, 23.142857142857146`}, {6., 
      8.45894710848252, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.03035818628132433, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 6.737273480334248, 27.571428571428577`}, {6., 
      8.119869376808609, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 9.632608824960474, 5.428571428571429}, {6., 
      9.510248317896666, 5.428571428571429}}]}, 
    {Arrowheads[{{0.04178652221351181, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 8.619891585573171, 23.142857142857146`}, {6., 
      10.52296555728397, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.04629832584403955, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 8.517151567980237, 27.571428571428577`}, {6., 
      10.625705574876907`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 11.775465967817617`, 5.428571428571429}, {6., 
      11.653105460753808`, 5.428571428571429}}]}, 
    {Arrowheads[{{0.01608718619303459, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 11.347958203640069`, 23.142857142857146`}, {6., 
      12.080613224931358`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0017036758500690048`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 11.675490655559216`, 27.571428571428577`}, {6., 
      11.753080773012211`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 13.91832311067476, 5.428571428571429}, {6., 
      13.79596260361095, 5.428571428571429}}]}, 
    {Arrowheads[{{0.00012532787862576175`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 13.859996746520437`, 23.142857142857146`}, {6., 
      13.854288967765275`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.00005696721755717861, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{6., 13.8558456346985, 27.571428571428577`}, {6., 
      13.858440079587211`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399283`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 3.204037396389047, 5.428571428571429}, {7., 
      3.081676889325239, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 5.34689453924619, 5.428571428571429}, {7., 
      5.224534032182381, 5.428571428571429}}]}, 
    {Arrowheads[{{0.0019012808859704754`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 5.329009084794642, 23.142857142857146`}, {7., 
      5.2424194866339295`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.0007743981136677382, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 5.30334840331724, 27.571428571428577`}, {7., 
      5.268080168111332, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 7.489751682103333, 5.428571428571429}, {7., 
      7.367391175039525, 5.428571428571429}}]}, 
    {Arrowheads[{{0.046264501558614994`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 6.3750646509494295`, 23.142857142857146`}, {7., 
      8.482078206193428, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.018843687432581677`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 6.999474566899553, 27.571428571428577`}, {7., 
      7.857668290243304, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 9.632608824960474, 5.428571428571429}, {7., 
      9.510248317896666, 5.428571428571429}}]}, 
    {Arrowheads[{{0.043475956259191616`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 8.581420832457761, 23.142857142857146`}, {7., 
      10.561436310399381`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.01770790353253569, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 9.168195082241027, 27.571428571428577`}, {7., 
      9.974662060616115, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 11.775465967817617`, 5.428571428571429}, {7., 
      11.653105460753808`, 5.428571428571429}}]}, 
    {Arrowheads[{{0.011787941493017002`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 11.445857959987505`, 23.142857142857146`}, {7., 
      11.982713468583922`, 23.142857142857146`}}]}, 
    {Arrowheads[{{0.004801268304739997, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 11.604954185147399`, 27.571428571428577`}, {7., 
      11.823617243424026`, 27.571428571428577`}}]}, 
    {Arrowheads[{{0.0026867163980399383`, 1.}}, Appearance -> "Projected"], 
     Arrow3DBox[{{7., 13.91832311067476, 5.428571428571429}, {7., 
      13.79596260361095, 5.428571428571429}}]}}},
  Axes->True,
  AxesLabel->{
    FormBox["\"x\"", TraditionalForm], 
    FormBox["\"y\"", TraditionalForm], 
    FormBox["\"z\"", TraditionalForm]},
  BoxRatios->{1, 1, 1},
  DisplayFunction->Identity,
  FaceGridsStyle->Automatic,
  Method->{
   "DefaultBoundaryStyle" -> Automatic, "TransparentPolygonMesh" -> True},
  PlotRange->{{-0.05427700344833464, 
   9.054277003448334}, {-0.05427700344833464, 
   17.054277003448334`}, {-0.05427700344833464, 33.054277003448334`}},
  PlotRangePadding->{
    Scaled[0.02], 
    Scaled[0.02], 
    Scaled[0.02]},
  Ticks->{Automatic, Automatic, Automatic},
  ViewPoint->{-2.4, 1.1, 1.}]], "Output"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["Run LBM method", "Section"],

Cell[BoxData[
 RowBox[{
  RowBox[{
   SubscriptBox["\[Omega]", "val"], "=", 
   RowBox[{"1", "/", "5"}]}], ";"}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", "gravity", " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    SubscriptBox["g", "val"], "=", 
    RowBox[{"{", 
     RowBox[{"0", ",", "0", ",", 
      RowBox[{"-", "0.1"}]}], "}"}]}], ";"}]}]], "Input"],

Cell[BoxData[
 RowBox[{
  RowBox[{
   SubscriptBox["t", "max"], "=", "128"}], ";"}]], "Input"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{
  RowBox[{
   SubscriptBox["t", "discr"], "=", 
   RowBox[{"Range", "[", 
    RowBox[{"0", ",", 
     SubscriptBox["t", "max"]}], "]"}]}], ";"}], "\[IndentingNewLine]", 
 RowBox[{"Length", "[", 
  SubscriptBox["t", "discr"], "]"}]}], "Input"],

Cell[BoxData["129"], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"?", "LatticeBoltzmann"}]], "Input"],

Cell[BoxData[
 StyleBox["\<\"LatticeBoltzmann[omega_Real, numsteps_Integer, gravity_List, \
f0_List, type0_List] runs a Lattice Boltzmann Method (LBM) simulation\"\>", 
  "MSG"]], "Print", "PrintUsage",
 CellTags->"Info3628000100-8197276"]
}, Open  ]],

Cell[BoxData[
 RowBox[{
  RowBox[{
   RowBox[{"{", 
    RowBox[{
     SubscriptBox["type", "evolv"], ",", 
     SubscriptBox["f", "evolv"], ",", 
     SubscriptBox["mass", "evolv"]}], "}"}], "=", 
   RowBox[{"LatticeBoltzmann", "[", 
    RowBox[{
     RowBox[{"N", "[", 
      SubscriptBox["\[Omega]", "val"], "]"}], ",", 
     RowBox[{"Length", "[", 
      SubscriptBox["t", "discr"], "]"}], ",", 
     RowBox[{"N", "[", 
      SubscriptBox["g", "val"], "]"}], ",", 
     RowBox[{"Flatten", "[", 
      SubscriptBox["f", "0"], "]"}], ",", 
     RowBox[{"Flatten", "[", 
      SubscriptBox["type", "0"], "]"}]}], "]"}]}], ";"}]], "Input"],

Cell[CellGroupData[{

Cell[BoxData[{
 RowBox[{"Dimensions", "[", 
  SubscriptBox["type", "evolv"], "]"}], "\[IndentingNewLine]", 
 RowBox[{"Dimensions", "[", 
  SubscriptBox["f", "evolv"], "]"}], "\[IndentingNewLine]", 
 RowBox[{"Dimensions", "[", 
  SubscriptBox["mass", "evolv"], "]"}]}], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"129", ",", "8", ",", "16", ",", "32"}], "}"}]], "Output"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"129", ",", "8", ",", "16", ",", "32", ",", "19"}], "}"}]], "Output"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"129", ",", "8", ",", "16", ",", "32"}], "}"}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", "example", " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   SubscriptBox["f", "evolv"], "\[LeftDoubleBracket]", 
   RowBox[{
    RowBox[{"-", "1"}], ",", "5", ",", "8", ",", "3"}], 
   "\[RightDoubleBracket]"}]}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{
  "0.3951781690120697`", ",", "0.06664974987506866`", ",", 
   "0.06574724614620209`", ",", "0.07098923623561859`", ",", 
   "0.06511762738227844`", ",", "0.06035301089286804`", ",", 
   "0.06832294911146164`", ",", "0.03412869572639465`", ",", 
   "0.03301653265953064`", ",", "0.0331198051571846`", ",", 
   "0.03291408345103264`", ",", "0.029778890311717987`", ",", 
   "0.03437609225511551`", ",", "0.029669051989912987`", ",", 
   "0.035268861800432205`", ",", "0.03330741450190544`", ",", 
   "0.034240979701280594`", ",", "0.02971157804131508`", ",", 
   "0.0340193472802639`"}], "}"}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{
    RowBox[{"total", " ", "mass", " ", "should", " ", "be", " ", "constant"}],
     ";", " ", 
    RowBox[{
    "deviation", " ", "due", " ", "to", " ", "leaked", " ", "mass"}]}], " ", 
   "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{"Table", "[", 
     RowBox[{
      RowBox[{"{", 
       RowBox[{"t", ",", 
        RowBox[{"Total", "[", 
         RowBox[{
          RowBox[{
           RowBox[{
            SubscriptBox["mass", "evolv"], "\[LeftDoubleBracket]", 
            RowBox[{"t", "+", "1"}], "\[RightDoubleBracket]"}], "-", 
           RowBox[{
            SubscriptBox["mass", "evolv"], "\[LeftDoubleBracket]", "1", 
            "\[RightDoubleBracket]"}]}], ",", 
          RowBox[{"{", 
           RowBox[{"1", ",", "3"}], "}"}]}], "]"}]}], "}"}], ",", 
      RowBox[{"{", 
       RowBox[{"t", ",", "0", ",", 
        SubscriptBox["t", "max"]}], "}"}]}], "]"}], ";"}], 
   "\[IndentingNewLine]", 
   RowBox[{"ListLogPlot", "[", 
    RowBox[{
     RowBox[{"Abs", "[", "%", "]"}], ",", 
     RowBox[{"AxesLabel", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{
       "\"\<t\>\"", ",", 
        "\"\<\[LeftBracketingBar]\[CapitalDelta]m\[RightBracketingBar]\>\""}],
        "}"}]}], ",", 
     RowBox[{"PlotStyle", "\[Rule]", 
      RowBox[{"{", 
       RowBox[{"Blue", ",", 
        RowBox[{"PointSize", "[", "Small", "]"}]}], "}"}]}]}], 
    "]"}]}]}]], "Input"],

Cell[BoxData[
 GraphicsBox[{{}, {{}, 
    {RGBColor[0, 0, 1], PointSize[Small], AbsoluteThickness[1.6], 
     PointBox[CompressedData["
1:eJxF0glMk3cYBvDCQGAjqKjLJjBxaYCqwwM5HGKf0oqcpQfQg5Zr4jUcatSp
i1tlg+oAReKGTpkxXsvchDkHJi6zSOJmPeIxQUStMBqnLoI6FWGFafj6vk2a
5tfne/r8vy+dXFSqKfYUiUSWl+9XnyOvPqn4SkWBb5rEJnyB4r9iFo7ujxDs
i67rjbmqELfHYFGljyEzKFzweCQltB0ObnH7LeR5Fd6SdrsdDOU7T09VB7od
ip5fxvY/Hu32u+i2Zu1qGggTLEbw4Cjd6juuUyMOgzbYsfFTcgRONPqFLiZP
wWtifUUpeRr8yld9aSVHYsK4k5E7yDNwIvdy1Q9HngqeCbPx410dg7cFz8LE
jPJn58lR8Bi8IWsPbJGOeDa+8C85dKXhmuBo7Iufsuk6OQbbkv8cukmORYFN
E3GHHIc5NUnTHeQ5WCqfdPwe+X0EHO4+G93aJTgeQ+LycbHkufB9dO5YDDkB
Kdi6JIo8DwfuhSbMIkuxXx5ov13+UDDgVzx8jmwBfvVeuOwW5TK8kbFsHucy
HN/j9HRQnoivfg9MJVsSUbP8a+VNyuX4/tj+4BuUy+G3VypN7OwTcgW87Kof
ZW5bFLBdPJLG+XxstVf5cD4fzdccaWRREmwJ5gbOk3ArLPET7i/AKslVbznl
C6A67R9OFiWjSLe2kq63JEP83LCe+ykwvVnUwb+fgmf1VXd5PxX9ruRMzlPx
0OW6C8rT8KK05yfO0+BTbHdwPx1eZ4vG8H463m5Tl/F+BvbPOxrEeQYkxrY4
755eIVfCc+nuWrJFifBl2gtelGeic/bePs4zUSqfvHkU5SrUJ5yfSYYKex5/
lky2qLBerztDtqlgtR/p4X01rjaluMhQo7MibQv31bg3rrCd+2pIXm8Y4r4G
yfcra3lfg/K+GgX3NVCYFBvoepsGkxbVDfD9aRHZ29VChhaXlQPryBYt2go7
w7mvffn/VtZxPws/V28Z5H4Wng/XHeB+Fj7aeD2L+1lYrlB78vPLRn175zY+
fzb29f5xn593NqJKKmO4nw3xe5DGnnkk9HNQ0TI8g4wceHeFy2LctuRgjW9z
AOW2HHwzLP+HcpEOjurtzmjq6xDn7/8h93UYOrRJSbbp0HrBkM59PeKtS6rI
0MN6cuJF7uvh2hDfz309UhovRfP5DdAeLIvi8xtQVVBURLYYkFTbNsB9A5rr
Sxy8b8TFKlcv7xsx1Wfzbt43wu4sC+H7N2Kl5rtpvJ+LkJAHGbyfi0WnS4p5
PxcezukTuJ8LZ8jnl3jfBLHzxXjum9A0585a7ptw8O+d67lvgixoVEAc9c14
vnJ1E/fNiG2VlHTUPRH6Zvy2TpJOtpnxb2DpB2RRHq59u8bnhtvIg3V75Fyy
5aWP7hxLtuVhRYT8P+7n44nTg69HPtI9+hO5nw/bqh0PeD8fK6Ytnkq5qACK
JYXhr/w/2gWJOw==
      "]]}, {}}, {}},
  AspectRatio->NCache[GoldenRatio^(-1), 0.6180339887498948],
  Axes->{True, True},
  AxesLabel->{
    FormBox["\"t\"", TraditionalForm], 
    FormBox[
    "\"\[LeftBracketingBar]\[CapitalDelta]m\[RightBracketingBar]\"", 
     TraditionalForm]},
  AxesOrigin->{0, -0.9675279956744001},
  CoordinatesToolOptions:>{"DisplayFunction" -> ({
      Part[#, 1], 
      Exp[
       Part[#, 2]]}& ), "CopiedValueFunction" -> ({
      Part[#, 1], 
      Exp[
       Part[#, 2]]}& )},
  DisplayFunction->Identity,
  Frame->{{False, False}, {False, False}},
  FrameLabel->{{None, None}, {None, None}},
  FrameTicks->{{
     Charting`ScaledTicks[{Log, Exp}], 
     Charting`ScaledFrameTicks[{Log, Exp}]}, {Automatic, Automatic}},
  GridLines->{None, None},
  GridLinesStyle->Directive[
    GrayLevel[0.5, 0.4]],
  Method->{},
  PlotRange->{{0, 128.}, {-0.9327327356660113, 1.224573384854085}},
  PlotRangeClipping->True,
  PlotRangePadding->{{
     Scaled[0.02], 
     Scaled[0.02]}, {
     Scaled[0.02], 
     Scaled[0.05]}},
  Ticks->FrontEndValueCache[{Automatic, 
     Charting`ScaledTicks[{Log, Exp}]}, {Automatic, {{-0.6931471805599453, 
       FormBox[
        TagBox[
         InterpretationBox["\"0.5\"", 0.5, AutoDelete -> True], NumberForm[#, {
           DirectedInfinity[1], 1.}]& ], TraditionalForm], {0.01, 0.}, {
        AbsoluteThickness[0.1]}}, {0., 
       FormBox["1", TraditionalForm], {0.01, 0.}, {
        AbsoluteThickness[0.1]}}, {0.6931471805599453, 
       FormBox["2", TraditionalForm], {0.01, 0.}, {
        AbsoluteThickness[0.1]}}, {-1.6094379124341003`, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {-1.2039728043259361`, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {-0.916290731874155, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {-0.5108256237659907, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {-0.35667494393873245`, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {-0.2231435513142097, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {-0.10536051565782628`, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {1.0986122886681098`, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {1.3862943611198906`, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}, {1.6094379124341003`, 
       FormBox[
        InterpretationBox[
         StyleBox[
          
          GraphicsBox[{}, ImageSize -> {0., 0.}, BaselinePosition -> 
           Baseline], "CacheGraphics" -> False], 
         Spacer[{0., 0.}]], TraditionalForm], {0.005, 0.}, {
        AbsoluteThickness[0.1]}}}}]]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", "density", " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{
     SubscriptBox["\[Rho]", "evolv"], "=", 
     RowBox[{"Table", "[", 
      RowBox[{
       RowBox[{"Density", "[", 
        RowBox[{
         SubscriptBox["f", "evolv"], "\[LeftDoubleBracket]", 
         RowBox[{
          RowBox[{"t", "+", "1"}], ",", "i", ",", "j", ",", "k"}], 
         "\[RightDoubleBracket]"}], "]"}], ",", 
       RowBox[{"{", 
        RowBox[{"t", ",", "0", ",", 
         SubscriptBox["t", "max"]}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{"i", ",", 
         SubscriptBox["dim", "1"]}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{"j", ",", 
         SubscriptBox["dim", "2"]}], "}"}], ",", 
       RowBox[{"{", 
        RowBox[{"k", ",", 
         SubscriptBox["dim", "3"]}], "}"}]}], "]"}]}], ";"}], 
   "\[IndentingNewLine]", 
   RowBox[{"Dimensions", "[", "%", "]"}]}]}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"129", ",", "8", ",", "16", ",", "32"}], "}"}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"velocity", " ", "field"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{
     SubscriptBox["vel", "evolv"], "=", 
     RowBox[{"Map", "[", 
      RowBox[{"Velocity", ",", 
       SubscriptBox["f", "evolv"], ",", 
       RowBox[{"{", "4", "}"}]}], "]"}]}], ";"}], "\[IndentingNewLine]", 
   RowBox[{"Dimensions", "[", "%", "]"}]}]}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"129", ",", "8", ",", "16", ",", "32", ",", "3"}], "}"}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"internal", " ", "energy"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{
   RowBox[{
    RowBox[{
     SubscriptBox["en", 
      RowBox[{"int", ",", "evolv"}]], "=", 
     RowBox[{"Map", "[", 
      RowBox[{"InternalEnergy", ",", 
       SubscriptBox["f", "evolv"], ",", 
       RowBox[{"{", "4", "}"}]}], "]"}]}], ";"}], "\[IndentingNewLine]", 
   RowBox[{"Dimensions", "[", "%", "]"}]}]}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"129", ",", "8", ",", "16", ",", "32"}], "}"}]], "Output"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["Visualize results", "Section"],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"Animate", "[", 
  RowBox[{
   RowBox[{"VisualizeTypeField", "[", 
    RowBox[{
     RowBox[{
      SubscriptBox["type", "evolv"], "\[LeftDoubleBracket]", 
      RowBox[{"t", "+", "1"}], "\[RightDoubleBracket]"}], ",", 
     RowBox[{"\"\<t: \>\"", "<>", 
      RowBox[{"ToString", "[", "t", "]"}]}]}], "]"}], ",", 
   RowBox[{"{", 
    RowBox[{"t", ",", "0", ",", 
     SubscriptBox["t", "max"], ",", "1"}], "}"}], ",", 
   RowBox[{"AnimationRunning", "\[Rule]", "False"}]}], "]"}]], "Input"],

Cell[BoxData[
 TagBox[
  StyleBox[
   DynamicModuleBox[{$CellContext`t$$ = 37, Typeset`show$$ = True, 
    Typeset`bookmarkList$$ = {}, Typeset`bookmarkMode$$ = "Menu", 
    Typeset`animator$$, Typeset`animvar$$ = 1, Typeset`name$$ = 
    "\"untitled\"", Typeset`specs$$ = {{
      Hold[$CellContext`t$$], 0, 128, 1}}, Typeset`size$$ = {
    231., {214., 218.}}, Typeset`update$$ = 0, Typeset`initDone$$, 
    Typeset`skipInitDone$$ = True, $CellContext`t$2121089$$ = 0}, 
    DynamicBox[Manipulate`ManipulateBoxes[
     1, StandardForm, "Variables" :> {$CellContext`t$$ = 0}, 
      "ControllerVariables" :> {
        Hold[$CellContext`t$$, $CellContext`t$2121089$$, 0]}, 
      "OtherVariables" :> {
       Typeset`show$$, Typeset`bookmarkList$$, Typeset`bookmarkMode$$, 
        Typeset`animator$$, Typeset`animvar$$, Typeset`name$$, 
        Typeset`specs$$, Typeset`size$$, Typeset`update$$, Typeset`initDone$$,
         Typeset`skipInitDone$$}, "Body" :> $CellContext`VisualizeTypeField[
        Part[
         Subscript[$CellContext`type, $CellContext`evolv], $CellContext`t$$ + 
         1], 
        StringJoin["t: ", 
         ToString[$CellContext`t$$]]], 
      "Specifications" :> {{$CellContext`t$$, 0, 128, 1, AnimationRunning -> 
         False, AppearanceElements -> {
          "ProgressSlider", "PlayPauseButton", "FasterSlowerButtons", 
           "DirectionButton"}}}, 
      "Options" :> {
       ControlType -> Animator, AppearanceElements -> None, DefaultBaseStyle -> 
        "Animate", DefaultLabelStyle -> "AnimateLabel", SynchronousUpdating -> 
        True, ShrinkingDelay -> 10.}, "DefaultOptions" :> {}],
     ImageSizeCache->{333., {251., 258.}},
     SingleEvaluation->True],
    Deinitialization:>None,
    DynamicModuleValues:>{},
    SynchronousInitialization->True,
    UnsavedVariables:>{Typeset`initDone$$},
    UntrackedVariables:>{Typeset`size$$}], "Animate",
   Deployed->True,
   StripOnInput->False],
  Manipulate`InterpretManipulate[1]]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{"Dimensions", "[", 
  SubscriptBox["mass", "evolv"], "]"}]], "Input"],

Cell[BoxData[
 RowBox[{"{", 
  RowBox[{"129", ",", "8", ",", "16", ",", "32"}], "}"}]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"cell", " ", "masses"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{"ListAnimate", "[", 
   RowBox[{
    RowBox[{"Table", "[", 
     RowBox[{
      RowBox[{"Visualize3DTable", "[", 
       RowBox[{
        SubscriptBox["mass", "evolv"], "\[LeftDoubleBracket]", 
        RowBox[{"t", "+", "1"}], "\[RightDoubleBracket]"}], "]"}], ",", 
      RowBox[{"{", 
       RowBox[{"t", ",", "0", ",", 
        SubscriptBox["t", "max"], ",", "1"}], "}"}]}], "]"}], ",", 
    RowBox[{"AnimationRunning", "\[Rule]", "False"}]}], "]"}]}]], "Input"],

Cell[BoxData[
 TagBox[
  StyleBox[
   DynamicModuleBox[{$CellContext`i12$$ = 34, Typeset`show$$ = True, 
    Typeset`bookmarkList$$ = {
    "\"min\"" :> {$CellContext`i12$$ = 1}, 
     "\"max\"" :> {$CellContext`i12$$ = 129}}, Typeset`bookmarkMode$$ = 
    "Menu", Typeset`animator$$, Typeset`animvar$$ = 1, Typeset`name$$ = 
    "\"untitled\"", Typeset`specs$$ = {{{
       Hold[$CellContext`i12$$], 1, ""}, 1, 129, 1}}, Typeset`size$$ = 
    Automatic, Typeset`update$$ = 0, Typeset`initDone$$, 
    Typeset`skipInitDone$$ = True, $CellContext`i12$2121340$$ = 0}, 
    PaneBox[
     PanelBox[
      DynamicWrapperBox[GridBox[{
         {
          ItemBox[
           ItemBox[
            TagBox[
             StyleBox[GridBox[{
                {"\<\"\"\>", 
                 AnimatorBox[Dynamic[$CellContext`i12$$], {1, 129, 1},
                  AnimationRate->Automatic,
                  
                  AppearanceElements->{
                   "ProgressSlider", "PlayPauseButton", "FasterSlowerButtons",
                     "DirectionButton"},
                  AutoAction->False,
                  DisplayAllSteps->True,
                  PausedTime->1.2790697674418605`]}
               },
               AutoDelete->False,
               
               GridBoxAlignment->{
                "Columns" -> {Right, {Left}}, "ColumnsIndexed" -> {}, 
                 "Rows" -> {{Baseline}}, "RowsIndexed" -> {}},
               
               GridBoxItemSize->{
                "Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}], 
              "ListAnimateLabel",
              StripOnInput->False],
             {"ControlArea", Top}],
            Alignment->{Automatic, Inherited},
            StripOnInput->False],
           Background->None,
           StripOnInput->False]},
         {
          ItemBox[
           TagBox[
            StyleBox[
             PaneBox[
              TagBox[
               PaneSelectorBox[{1->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJzt3MENgjAUBuAmXjw6wnMSd3AEE8+u7AiMIFcM1RYItfH7EtILf5sSSlMO
73x7XO+HlFKM12m8jgkAAACAhiKl52Wm7SUP/ygy6+a93TE/NB7fd4TuxML3
Npf71s/MfUOm/cn82vmX9gN7iO3e26L990Oudv3mcpN26/HXzr+gH+uf3YT9
3/4PlcL53/kfVgr//wEAAAAAAAAAAAAAAAAAAAAAKBCFdZ8iU0dGXl6+3/pp
UVh3NvL1Z+Xl5RfmW4vOn5+8fM95gCov4jQcbQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 2->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJzt3C9oG2EYx/GDURjUrGYq4hZVs5mY1TQnajZTGFTNHIO5QslMVCFURcWs
JqonVnM1UzOpuKmYzsw0/ZP0mtR0G9tYq6Z2fe99wuW4e3Mw6NLx/UB4eHLv
k4TS33uXiHv0auPF63uWZdnR40H0uG8BAAAAAADgH3IWFs6rUQ3d+WE10Rfl
rn8c3KxvXL5V1dF9YfudE/X+D9ePqokeQD6v1I1z6/vDrCrH89g/vg+SeQ/a
7fOJ/L8pnRWZH1fL+qnmdV94XvK+5x9P5H/KvPd+Tq3zms1eZtXHgVkW5SDO
7fWzUTXRT9OoVCbXP+kNJ/o9P/N1GmtrKt+urrbObbo6rVGY+Tl0LqPPe2aa
D+r1zPx6Kyun6v11zZtv9Lb6pr9DtF8d6f1K1UZr1Ev20T54XOTvKOvd3+8O
k/PArdjvjHQeLqqJfhrJvyv7gM6N9Hn5l+fdcnloyl+orwfSouOhaS5d0ySX
QSc+z+fOfTo4LZJ/T1eZk75o/sfXC/X6YfL6AbgN0f+dymu49fiimuinkfwH
er2z1Jro8/JvD/rx94Z+33j+t7eXM/Pvdmsq/8HcS+M+EO1nmdcP0fWHOq9L
zZsPR0+NvyNI/u1yWeXc2V09TvaFz/+yXv/+ID0w0xY3h/o8OsqqcjxP0GyG
Oi8T3xekl5xPm/fkdeT8q/ui89agr/aB8CA+30f7Ur/IvNOtqfWBXE+kqhwH
kM/Reff07w/SF2U/34nzPv8h3gd0X5jOv7d8eZLcDwAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAs8XduY7vS7e7OjTV8bqUQO4rJ/eZyalBzv3nmGf+Ls9b
28vxfQ43Px+aqqybNaHv/1L5rlSuTFXWpdkF71tr59y/lnnm7/J84Ptfbp73
2u1LU5V1M2dxU+U6rJWuTFXWpTndmtyn0lhlHfPM/2fz39TzS62vpjpeBwB/
4w+OLNuZ
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 3->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnH9MVWUYxw83G3ZpohXLZuKhms4/UkYbs7XkTMuN1lwZSzdre2WDWs3l
j6C54TgUNPuppbJrMTnMQru5nKLGjLWTTPwxnAJX9ILA4bd4gYtIumVw8z4v
Lzv3cjq7LlZ36/vZ7r68z3me9xzgft/znPPHm5T53qqs+yRJku9+Zt79TJcA
AAAAAAAAAAAAwH+IdjDuSlrwh5TUbtIRJx9HSn9NWzBfdztIjeP9bfdSL28u
ukj5NQFPUNUThRfv6fwA/B+ZM9ZMvsse7Qmqknm9i/w39i2NjTkLmu18pCYc
N+h4stERVG35vM6gynO/pLE875NWWx9uOnWVjj/vaxnPGwoq2/0CjY1Dribb
88c/ytcLb/pl0l0byf+Ku6qR5tn7TotdPfu+tI7y6m410PVXJvLxWS/No720
ox7rCIh22Mou8pF27sNr5IMVPVcj+t7ujaN8llzYS3qgivveeZBUWuPutJzn
qVF+f3eu7jT7VqhSccdP92G9tduq3lhxmt/vC2pbzXWymCcmhtRYet6wqtee
LSCf6q7FTZbnL3EM8eO6Zf3EPCV83ZCzniGfq0td1Dfoa2v5OnDyoYj6GEP/
qJby2wtoHnYstxbrBvi3YLmzrpP/3Q/7yDfNhX2RfP+0rEXc71t/7TX7R9Va
KM7m72u3moc5UsjXuu/BLrNvhUo/DZP/ta3Ley2vY9syihuBgN+qXp4+jV9H
n2PIql5pbCBf6q80XraqV133k2pxlbb9A8v+jO7z0vzf682/PysvprFx5NVL
kfwd1dh63i+U5VCdnFQQUR0AUwGLj6f7KWsp5/6PnRXZ83PS17xf8GXQeqEM
/0JjPf99UmlVv+X9X9nsb+L9+WPdlv4fuUW+1l/Os/S//sMj9HxgvJnZblnv
H6V69ta7ludnZxZQv66UsxareuM2XzfUn1/ssPV/cS2tH/qYwf16KZ1raR6p
WrEysvcYLP8C5a9P5vNlbLoA/4Nox8iuof6b7TpG/YPyxnbyvZaQSGOpNd/+
+Tlwh/vzdj71C2r1dlL5ygjvD4pes+wfBKpzAx3X5+7uCPFx2gwaqzNibN8f
yEW8TtnpojzjCwc997DZR+m6de9+2/5f3XKOP/fnnPCS7iij9wbqHxtpfWPF
n3rgYwDsMer89J5QS/2A+ggtfcj2vWE4SqCa1hHtuQG+nkw7bP3e4e/Or58i
H8up1/j7zCU1DfAtAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAARB9aXCntO6X8uY7vP3WmIkRFXOSFozi/4vtO3/DQvlOysbrN
rCIu8sIxjhyl/aXktX2075S+Z0+IirjIQz3qp7JeLWONfL8mxvc7fGCJx6wi
LvImnf9pvs+69voGvn9aYlp9iI7HRV60oaZ8N0y/3zYvqZ675qZZRVzkhcOq
B2/Q/nBVCXScDXwcqiI+nheOXvg235/2mzy+32xFqIq4yEM96qeyXj3gHqB4
5U5SJfBjv1lFfCIvnHVP8rx9C0lV1yKfWUV8Ii/KUGoC5E8tYxn5XYpvCFER
F3mTcJ3k68bCQa6Lc26aVRuPi7xJlPj4/+m3WFof1J7HQ1TERR7qUT+V9fL6
8+RrbYuHVJ45O0Qn4uN54WhnR7jvPx/k68bpJ0JUxEUeAAD8I/4CHkhVvA==

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 4->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
qu+3KU6osAs/hmGYf8VfPwB7vg==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 5->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 6->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 7->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 8->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
rQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 9->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
2x8sdw==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 10->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
+qeMpPx2/fKpIuK68DEMwzwSfwNfFzHQ
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 11->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnAtQFWUUxzfJV1raQ0vzsZhpSm8txxL81HQGKTPLR/SY1SQ01DJDRG34
mkxDrBRF8r1g+MwxrURDba1UKETFhGtofmAoYb4t08yKc84y3L3bHbAmnTy/
GeY/9+z/7Ld7755vv90ZTuCgl/tEBGiapv/1V/+vv1oawzAMwzAMwzAMwzAM
cxkg36+zs/OlPgiGYaqESs+zyurWatxMlalq2XTDpahjY1rrHTx/MMxFEt9n
d5Xq57b1ReA/+ep+qP+EuB9AR08pLFOz/fiCyuzPGH4ut8ynez4ohfmjYdEu
+BxVb1dl8s1b27xe0We0bbHg35gH9IRB2f9kPyoh9Euej5jKIqrJlLLrRdaM
jK7KdSMbHYb6U7l1D4K2/2xPmRq7M1/ztx9rxENQX0qOPQL1Nts8BnU76cbD
oKWTM9zy9bC4JWVxERFzAOr93OjjMF7tqaAyqB6o/mTsUbd89cowmBfUziUw
f5hNh2Dexp6ofT5BXVZ4yDU/tAvc541VOVCfZsy+b8HfMLcU/Wdgv+LBjfmV
+R7FmpSPwFcj6sXPy/azPnr6xdStvja2avMnIXeLz3ieuHIxvg+C31+ObYzr
6PQm6XB9/xb2pr/rQs/f9TVc/+NCoF7EVQNBzbfqYj0elBtd80fesBp8HTtA
vehnq2P+6WtQe8yHeUC/0OGAW741fAvMG7L3RBwvzARV78zAz2d/hnwj4YXj
bvnycDDGx09A/7quJ+C8QwfDZyvkDtT5rVzztZ+PwDpDhMTD+wIReRR9m8Nh
P2ZKLM4f07eX+K2rk1vWwHFObg3rFbF6OHwfZn4TeI6R2oy8/7IuZUEhP79c
gah9oclw/d2ybTP8/gVnU/xeB0mzkuC6DY6B60Vf9wneh7P6Y90MCIL1gLip
+ibX+pu+4FPIe28yzhMFTbBeNtyPuhXvx0oMK3W9/w49CPdb7dsNWOcLR4GK
ftWwDhvEw31f7bj2hGt+8Bxc7z/aHf3PBuC8M2ge7Ed7qy+oVRTiWv8qcSB8
T0ZbXK8YzbriemNJNJ5PYiHUvZXY+pi/71FZL30FeRlpsN7RrvIUQ/0vbAWf
9bnNKvX8YWMVn7ok7z2Y/wf6F+cXV2nd/0wCrFut8EwP6NRkULU5DdbF1pEH
Zrvur1/NaVDnB26GdbieFoF11qkF1mH7+lhPr9Te7jp/aEth/pCld0F9qbQG
mF8YAJ/lJswXpbrr/Vf2+ngt1Gm/++A+a97TDeu0fzbmt/0D6//mJP/1l7Uo
B7bfVgPHexqfHyz5GK4rthdt9Vv/NV6E9wTWm9uXw7jdU/G9xS+zJ8HxxcYk
cT0zlzv6ec8KuE7f3v8GXL+DS+dV6rn3ucwPoc6HjNwGealP4PxRtHElaO1O
fvcjw/fC84Uedj/kGc/XwXkjZx6sm+WqyIWVOQ7Z6AS8tzDNUFx3vHGv/3W7
ndfkQ3gPYbZZDcdhHcrE55X8FfF4PusSK7MfPWd9WkWfar5vBtc9c6Whcuak
Xsx1b45JxPvkgkFz4T7c6t3kquxHdZ4D7znE3LmwLlF3tvP73uNv97OtTQLX
LcMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMw
zJWDPJ73Hfy/+vKu2MeypA+o+BHVoLhBPifiqUDolyGi06A/pdF7DfaZI9Xt
OPl8WNEJ+kyae6egxkRAPy3DVorbPif6+xL66pgB14NaQfV2VlTtaoxr5HNi
rhwI45hZw8Ev63aDPkGCVFJckM8nP605bNdzO0L/MvNMAfbZ/ZWU4hr5nBiv
Dobt6t3qMJ4ZGfgNjEdqUFwjn8/5r47A7T0mQN8TGTYrC/uYoMruGDdXRbj2
P9KSAqHvkJzREY7X2L0lE74/UkVxNR19Tqxej8B2ca4d9G8UT52HPkWStDxO
Pp/vr/NQPN7s2eA3uzQCVQJVfINxQT6f8U9PhP5p+vKR0P9MJF74AvYzDdVa
hnFFPidifBz2OcvPgT5Q2tBxGRXV9GC83OfMn3AX9LlUA4Kgb4sZOuRj8PVE
lRSX5LvcEBMbn4TfvVZvUHPSGC+147bPiTr9K/TFEw2vQ9/SbphHqlHc9jkx
FmdBXG1OB5VxP3qpRXHb55MfvAz7ZPb6CXVmtZNeSnHb50QOWIDj7CvG7WcU
qE5qUlyQz2f8hql4vJF56MvbhvmkBsVtnxMr6jWMN1+CWj0F80ntuGn7nMcf
EIXbj7yHWjLTWykuyOdEHQjG888Ygdp3Ef4epOVx8vmc/0sP4jjZ/XGcPRF4
/KSK4pJ8PgzEfqfG1rvRN7IOjmcrxW2fz/jFhdgfbU8g5gfE4WdSg+K2z4e3
Pdgn8oUo7Pc0EftFClI7riZ5XPNFyDgcr2Uybs9LxX5vpAbFdfJdblgf/Q51
ojdoeQrOe1O4l9px2+fE9NSA7caOa9H3S6SXSorbPidGdmvaXhPH1TFPNaf9
UFySz4nc34iONwyP4/Zp3kpx2+eTH7MfzsuaXwfHjXoAfcNQFcVN8vkc/+AS
nPeK/8DtPYIwrzuqpLggnxP96gzc/nQ+bh9f+1RFteOCfE6sW9biPJe8FTX2
mlNeSnFFPp/xHx4Fcdk+HrXA460U18jnxNRj8PiMseivvwXnf1I7bpHPiVjV
GeNFqCo3Gv2kOsXLfQ5k10CMPzMCfSVJXqrCMV7uc35/5z7H+SXwIM5bj+O8
b/VSXnFp+5wcX4rzXDCqnl7opRbF1bGl7vkMwzBV4U8J0G9j
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 12->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 13->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
brBPEAThb/EHJixPyg==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 14->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
ivUncDbWiQ77sQ5KteOG9DEMw/wj/gKh4l9t
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 15->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
7M+pXZGfbOK4Wd24xj5BEIQfxDfIQlVB
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 16->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnAlsFVUUhodgWcQFWcRKGgfFaFFWwQUxjEiBgoiAgBCEoQVRBMUIqay9
tmBka2UpqIUyhE1ZgyBlq1ylUEqRFqgslsJUSi1CKdAYKwpIz3/H8OZNmgJR
kJ4vaf68c/9z5955c+6895q59cLe6zaooqZp+pW/6lf+qmgMwzAMwzAMwzAM
wzDMLYA50tjc+mYPgmGY68Jeu3jrTa3fAQMlrx8M898gmkRvup3rzY4ITruR
+RkVmvF6xJQflg3NLbnezeKw+SVqFU/5tjxd/7Lz1G3lab7MrYF1uHpmyXVn
XDqRXqL6oMtZZbkO7SFBayjvrjuOk//lu3+h/AeOHC5Ru1fhmtL6EaG9dpS0
S3PSIcr7dn8+1X9wbiHlhccf9sqXRuA+r7j+6KSUkrjoXu1H0oEndngev9+F
m/s941/CbHPq4O04r/KCmbFcXMv7Z1wYuWjrFRWxT1K92ournKI62tX2eGn9
WJ+2mliSJ8cdzaF6ey6kgPIurjpLeXPS6bWV18zzejK6LFxLdT9Tnqb2nD8o
z5wTcY76CQ2l1yIpuMAr317eA/ENoci/pz3lya2DKc96OJtUJqw/6zmPpKo0
bpl71iZ/QTitG+LhxjR/I6oW1o+stnmlnQdzWATWi3ZHV1E/6RXpvBkpKw5Q
XkT80TK9Hx+kzL7ap7fftZ7rkCkr+oh0un6sBzJwHSVlzC71+sm33qQ6u1Rz
L+WtCM6m6/jsKKoX3WqVDY1O8urH2vQ6XZ9W9Cz4l29H/TYcgPqbF4N6jIzy
rB/74OZjVCeDZ5FPW/UJ5RliNeq+/lyqP9n/lXOe60dG0c/kX5OD46+X0NGN
yG937Uv5ZqUsz/rXF42m+ZmtF6B901uUZ2mpmMeI1ZRvy06e+Xb3cx+R7923
qd5FJNYvq3IBqR07BOMYvS231PfhyMCQq9tF9Hhah425/d6n/h5q+x2vA0xZ
Mf98LOGa7v8Ze+n6MqJj6D4ostNwX0yugM/v86uu8Kz/ZxfT/VvP/gbXd8Hn
qNuQQNx3syvj9YITRzzz4+5NpvbDBur2iTeQNzQAdTdvJdaD1GG/etbfo232
U/0N6oD8iOdwv67zAupvTir6iQ/yrF8RJi1qD389F+PeQ37r90VYN2qq409f
eKzU83k0gdYhcaAFfQ4xajyG47X9k9YFc3JxdlneDyNnL31/kt/0WUfzmdwl
g+ue+c84l0jfc43+o6gujd3TV5Xl+hOFX+yh6//iDqoDM3/0CdKsF+nzrznx
h5We9RtTewPV35YKVF/6oa55+DzRAOuGDMTveBcOpXrev9+KW4j7b/huyhv7
JOr9zkLUb5P6WAca55b6e5idUEjjtydEY/0IbkeqV29C3y+M5xfsKct50GtP
pnq1N96H81A1KfN66tdo//TXXPfM/xXzrmnX9f8mObsQ3zOK8vbh83fo99fS
jzkoBL/TXXyHfreTQ8bQa/FizISy9CN/fo3Gbbd4l74PiX39r+t3dH3UGf7e
zjAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMUw6R
PRr9RM+nfznmJ+y/keOjporrPeHzY2lv2i/PCpC0P5f1RKuD2IcDKlTc8bkx
+sTR/ht63BbswxE4du/V6sQdnxvr41jyGfc3IpUNVtDz85ZSrQ7iQvnc2G0S
4cs8Sc/ny9ktsQ+BUms/4vpLiZ77aMheUfDXmkb7B8gJDdOwXwFUc+I9o7yf
/98wDu0t+u6i9ve/2Ek6HCqeRtxIhM/v/K2Lonar0qvkt8ePwPPKSq0AxMVa
+NyYr8xAe0QH7H+4xKJ9G8ylUO1DxK3O8Pnl252Q17+YnnMW8Q+S6kotFdeV
z42IjKZ2s+gNek5bD0oktetCzfOIi/HRns9RiyhJ+87I2I60P6veZtpG0tZQ
IwZxQ/nc6NXGLqH+x/WkfVzMod2mk38kVBul4lXh85t/5+20z4q1Uv+K9GTC
TMrPgZpfq3gn+G41rCqZ2F9rbjqpOfY77F/lqIo7Pjei7uPnab4tWpGa7zX3
UU3FLeXzO35mDYrrYQ3RT98GpEKpE3d8buxLp7HPX+4xUuPAcbxWqqm4fvm0
5/itpqloz9uJ9pAUnAelhoo7Pjf6RPhERDZ8cYeQr9RWcUP53MjL2xFfDb82
LAv5Sp245fjc86/8Fdo/k2gPSvNRqeJS+dwYF5dh/hXht8Olj2oq7vjcmM8u
x/z7HUReAs6XUGqquKF8fsc/MgF5XbbCN2YL8pX+E1c+v/MXPhj+nUuxf1Ny
EvZbUmqquOPzYxv2abPXLIMWbEY/Zzb7xB2fG3tbbxy/odrnqWs+fEqduJbc
2/v4Nxk99u0iGucuqDllvI9aKm7HQN3YcyMpbnSMQl7tab6q4tq8SM98Pakf
4jMGop/dw3G8tOE+cdPxuRCTnqG43qkpfKEtMW6ldkfEpfK5MZLrIq97IPp5
pLmPChV3fG6soGK1vl3G+vVSHeQrlSru+NzIgEro97UA5DUPwrifgsruiNuO
zz3++09Tv1L7Devu9poYr1InLpTP7/iFfyH+ZgUcP76er6q45fjc+V/mIj61
CMfXcJ6EUnsK4rbjc5OVgfFX/wW+dxpjnkOhtorryuc3/6cW4/wm7kE/x2ph
vI6quKl8fvn5LdG/HIn2lByMY0eOT1xXPr/8dfXQf48wvP8Z6fArFSquKx/D
MMwN8Te6zFy6
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 17->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
bYd9dkdygLpxldLWP18QBKEi/A5/iyS5
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 18->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnAtQFlUUx78yUrMmHM0HmK02lU9scsw0jesj04FKSXr5YFOhMYlE1FHT
XBXLsWBKQYtKFzVNS5y0iIfK9REoEuYLC6U2wyeKij18FBbnf7eJ/XYYwKnJ
4fxmnP/s2f+593zLnvvtV7O3zahXQsLreTwe7a9/vn/9a+BhGIZhGIZhGIZh
GIZhGIZhGIZhGI9I8c0LvIZ8mZi4+VryGeZ6wlj64p66fL+LHiWZdfnzM3Ub
7aUZORX3v37r+oKq+kC0z9/CfeKNlZRayNfl+kU+7r+6Nn8/vUPj9Io8LcDv
2woV7QMOVGccMXfpO9RvxfPJr83oXETHEywax1NQsraqcWT7y/R9Le+XxZQX
nHmCjo93P011rHrqkFu+vizxCJ1vuHsa5Y0djvlvHEPjmPl6KY0z44NjbvlG
XsJe17o6xQzLqhg38IGDlP/DvnXX0g9i4tTl3E/Mv41YH2pSP2wduITu26cb
J1XnvtN73vYd9c2Je6ifZHrmuQo1fs6nPhJpq7e69k/p2EUVfWJs2X+Wzi8J
PI+87qTCfyiptTb7uFu+bHv4J/IHpKBPVwjye6b0LqN+jpYYt3XcObd8s8Gc
U5TfKATnA39F3btzcdwvidSUP7nmy3bHqL+1c+uoPnFqOK0TVhTG0XrNLKHj
5s1c8/++Dlk3bqD6d83C747Wj9O6p1+8j66nR+th1aT/jfRRiVk18DN1GyNi
1gK6v2a+u4zUL5T+e4/hs/D9Ku+7tFcT6PxkSfen2V3SfS7D55LquVHUf3pq
yC7X/rmlKz1nW7veQd5NGeQ3x/SjPpbBS0j1dc+fdsu3OiVQ/wnPW1g3Li/C
vFMeo2Nt8STM37Ttedf15+qlw3R+w+/o+27BmK/eTejfR1CPiFzg3v9xF3dQ
vXlBZ8g//huMY2IcT+fzyC/4+GyVzy++02n90NIG0+c0G7dDHbcbtJ6YP5QW
V6v/I0dUfh7xj/2GnxuY6qJF75lD90v+abMm9401aFoy9fHcFidJu0p6/tYL
fX6kceY3i3Xtv6ZXab0xi9tS/xgxn1D/WAl4HhD9g9BPvj5FbvnizjS6v80N
r2D9GH8OeUNGYR14NBR97Fd21DU/ZhutS/rsXNTtOUr5cu+bGGf2DOpDc+XF
I1VdD+uD+uhPGYv6M5dh/uU41nMGfV/l9/+JxpI+58ISzJM0lZ5LdBFDdYs2
t62qyd9DHJm4k/ue+a+xVuasxXP7A9l0/519dV517kOtYA09/1rmdPr+MsOO
Y/04lfE19WP0sPmu378doz6h85Gl+H29aTN9XxpaAp4j4n/D74bYMclV9m8D
nfrF2B9CfWfuGYzv68wnaV0wg+/KqfJzbOtK/auVZ+H3jjmJ1jOzw3g6No4+
Uq3/f2aEBdM4ImM0Pdd4Wo3M4j5m6hpaVtK02tz3+uW8yZRXkke/Z7RmrWNq
9Byz6mocrUPv11+I3zX58bWpw5p78G3uW4ZhGIZhGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGKbWnCijfWa0iNPYZ+uLA6Tyc6hQcV35nJg3
l9J+WSK8N/b5OtprP73vWgz1qLilfE6sbjfQfjd6vxX0Hr+l+dP7+EYbqB03
lc+JHtYwn/xRw2l/W+PnAfQer/aL0vGIS+VzIoPKaR4xsiHel1/Qm95b1pRa
IxAXweWu+5eYU33ovBX/fDb2/ei3jepOhhoqbk7xcX0fXx/Un94rlj0f3E6+
RSnYByBR6cOIGwP7u75/bHSZ/BXFe0yk/ZWs0cYmGkep9hDiZoDyOfO/3Eh1
irs30nxW2s4MvH8MlSouU+Hzyg9/g/Y/NHzmbaS8SYVf0rGtKu5RPifW6kCq
0ygqT6PzQw6m0vETUHEIce1j+JyIOWdovzk5sojeHzd/iVtDn+MsVAxH3PZ5
1b8jdDHlt9uK99T9Go2j+YdC5R2IG9vhcyL/GED75Ol9Zifi+k1YRHkpUHMQ
4uLKgFrtp/dvY8Z3pX2yxLxnsV/Wa0NJDaV23PY5ES82pbj1eQB87TpVUkPF
NeXzmn9CB4rLaX1wvrwH/EotFbd9XvnnA5F/OQb+5Ej4ldpx2+dVf2FPfE7/
UGjKSPjWQXU7rnxONN8y7E8WpmE+v26VVNMRF8rnRL9yBvG1rZAXcC/m6QKV
nyJu2D4nEScxf4u2qLfsPsyrVKq47fO6flFFiP/eHHU+Ab+h1LqC+N8+B9Yz
b2N/lJws7Ff0ei58SoWK2z6v+RvNhK/+GuzXNEvlKdVUXCqfEy3qZfiWnsL8
FxuhfqVyCeK2z6v+vRGov1c6dBLUUipV3LMvwj0/0hf+CxMx/4PTMZ9SO277
/m/oLY0L9LlfiCMVgctIDaWWimvK55XfZhbykj9EXnoGxlOq2XHlc2LtzISv
4y74nvsN4ym14yI30zVfu/QZ5vt2N+o9UAZfAVRXcV35vFi+HvnRX8DnUwyf
Ujtu+5wYQdGo7+t4zN/3K1Kp1I4L5XNiZqt4K1xnY/d2+JVa/ohb2e75otdQ
1B2G89bLH1VSO64rn1f9TUJwXkRi/ndX4NhWO658TrSgJvg7Xeqr6oxF/TlK
Vdz2OdE/agHfew1wvTejTkupruK2z2v+cYVYpzruQ78/53fhn2qquO3zym85
H+te3HLkRxVWUkPFPcrnxAwfB9+Wl5CXuhL1KNVV3FA+hmGYa+JPUgJN+A==

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 19->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
vC4leE4cPBf9a3XjpvYxDMOcFn8BF/RVYg==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 20->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnAtQFlUUxzc1QSzxVSBCrY1O02N0emjac3vZZJoTlqQ2upHPEtOxAtOY
az5KqVCz0kpciSw1LCQzjcdVTAVRS0OECi5C5gMDSafH9OT87zay3843TtaM
1vnNMP/5zv7PvXeXe+7u983s7RL/eOyo5oZhmH/+tf3zL9RgGIZhGIZhGIZh
GIZhGIZhGIZhGMPMKSu+5TTy1ZTmuaeTzzBnFVNXf8bznWEYhmHOTtS4vNJg
93Fr6UOfB73PP/Aq5dtvxWzw81mRl37KzwnMfw11Q4/X8v9Gnt39uUzKyy79
prEuRGQc1ZdUtR8FqxPn2457G49b78ryRjUbzq1uVDW/ZD/FZ4ov/PLNhJeQ
N6LXukZ1MlKRN0AeoHbq768lHdj9W9/+72ugccqkUQXUzgXdMd6+kvqVRVfU
U7tp7av98sW2WQW+4wqfs4Xi6y4Z2Xg95Gs91emsE8551gpeZ5h/C7s6/B2a
/3vD1tC8L5sjqY4uu55U5LV/J9j8k1Xjauj4/O0NlH+knOpG9u52jPLv7vul
X7796mNbqb6+G0d+lVoBv5lJag2uo7ixcm69b35yN6pTZSSRX6WsQr0evoU+
G6sTvqN25izzzTc2FWGdiMij42ZiFPKun432tkbVkTbr79//gv7FWDeex7rT
IYbWGfVB1VE6/7E7qX9lLffv371+a8/bi+M19LuDeLGArqfZZyueX1qH+K4/
DPNPIEoL8Hz7cNZu3G8X4T547/uFNB9rw3zvPyJ6Mv0ubG5YTfNVhDZD3YxJ
JXVGV1D9yENdy3yfn++NQT8tUlEfd4XAv/IqyjfLdkJDIo/63mdDU6qov46J
5LNXRFM79nPpqOcDM6g9K3XEMd/6qVJ7KH/JfqxXRb+Q3yn/BPlxQ7EuDent
nx+RkE55g7JQ7y2n47yzFPLXv05qqYrDQZ9/ql6g5xszeRI9r9jjl5JfdLoQ
1+Oe5vx7JPOvY4t522ierepCqi49mHEq806c6EX3L+fr0Ydovt9RUov5W0n3
LSum0L+drA2bad4vmUl5TvOduH+/NBb1U3kI9Xj1sRLf9ad11Haql8IVeL6f
dwHyfojDehIxDc8TX7X7yrf/8HEf0/gS6+l7gHljO+SNb4b14Pck1G9adNDn
d7surJKO7xqI9SLtYaxD97XF5/xBvuufi7xn1xI6z8gS8omokIOUn/fz15S/
PPl1rn/mTEf9aqVQvZzzyA6av1vKpp3S+pG1cS7V28giev6Qw+bie/+CyXR/
tsJ/nB70/tkzA99XfkOeuHUH1Y/avZLqVu17JD9o/Q2opnzzyFX0PGG230/f
C8QbD1TQeJbNCvr7hd0uxaH6jRhGzzOiLJ7GYa5bTp9VTm7aqVwHNW3QMjrf
B4vpd0XnUNclXPcMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzD/P+TMbNpfQibqfSZCF+FzCNRI0vFZ2b77aNmdyum9devJXHpPz7xd
0Pt79m1QQ8cd7QvgroJd5J9XvZP6fXc97QeiVkCFjou+8HkR2bfT+78yfgD2
KSi9gt6bE/ug1kjErQ/hC+BEK+RlTII/evFGGq9WS8el9nmxelu0z48aGU77
AIkRY2g/BGlDLR03r4PPi+qWQO8/Gx3aUH/G2wfofWQzHSp1XHbVPm//2/I2
Ub8HMvNo3LmPrz1ZVQ3i1lb4vNivtEW/g2M/IT2cQ/u/WEegKg5xZ6H2eRBh
23Pxnvdwej/SnNPyA/p/ztWq41ar7b77B5u7F9K+TapzLPXnTLwyi/wToGYn
xM3P4AvIv/M97FfzwtBMauf9VfSeuZkJlbMQF9oXQOnxBIp33pOc35gfO5ze
s7YGQu0eiNvZ2ue9ftlJb1B8+IZnSGNGL6BxdISK8YirTO07w5CiBe3XY0/u
RGos6ttELR13fV6sGTdR3LnkKbQTN72p6rihfQH9f96L4uaa2fDf/GYTdeNK
+wL6vykK8T1jSUXCFORpdeO26/Py2DUYf/R4nOfxJLRzAip13PV5cX66Edcp
foYe93zka7V13PV5EUcuR/yiSehvAa6b8TJUxiD+l8+D3boO+5KUdUY/I+Cz
tap9iFuuz5s/thL7lUxtA1+0iXytQsddX8D5F1Qj/6frcN3PwbxRBtTWcVv7
vMjBtRSXT3yP/Z56dGg4WR0dF9rnxaopxP4wW8rRjhGJ/Wdc/RRxQ/u8mP02
ov29fXB81W/YP0Kro+OuL2D8xSnoN6wYvkcL609WR8cd7TvTkOmLv8f6lwHd
00BqarV1XLg+L7+upLizNB++0KNNVOq46/OipmxCf7d9A39UxHFqT6sbN5/e
5J9/7XrEhx4mteTFyNPqDEHc7LneP79qF8Z/ohbjHnU55RlahY67voD8RJyf
Pb0CvhadKU9qdeNS+7w4Y9ag38Fo3259Psav1dZxpX0B/U9IRnxzGs5jvmqi
qgBx6fo82JUT4Hs2FRpdgusVAzVmIC60z4vVJx7XfcdsHK8vwnlodeNC+wLy
j07C8dgk5NXkYPxapY67voDrF9sP4506Ee00w3y1tcqnJ+p2+vnmy5z9WFfj
W6KfNjba02rruHJ93vz0zVjv6g5inVvaAePQatQj7mgfwzDMafEHanw8jg==

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 21->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnA1QFVUUx1+aDSWSTR9aVqylZmmZRlo5xa2kLK0ctA9N65pGkB+hTVpq
umJpomWo+dXYbKCGo4DZaCkhFy2NDEQJCMxxRYo0n2BmYV/kO/+7jezbXqZj
U835zTD/eWf/5967yz13972Zu62feCb2ycY+n8849tf82F+Yj2EYhmEYhmEY
hmEYhmEYhmEYhmF84mjnguhTyLfqYz88lXyGYRiGYRiG+SdQ3XI/CvncuuKD
LaGOG11X5oU6bkeKHH4uZv6vWM0eTz6Z+W0vyykK5BkRdYtzj6nqlzTnb7Uj
ws+j/Ecu30h5zW/wHIdKOyf/+Lh8qNVyyosP3xBQdXNEVUBl3/oKz/xrCnZ7
xrPesQJx2xz9NR1fufwzz/FfesUHJ3Jedt6sTbxOMP8VzIyY9EDdGs8//TbN
25RRw0LNX/FhyZeB49Zdz++j+vt1eE1Azajb/RTvH+ZZf1bfGbi/Volq8k9a
SXn2kOpDVLe33F8bUBFxS61nfsoKql/72r41GGcE+cyfp1G+ceNUUit6Qo1X
vtlq2be0Tky4Yi/51v7wFfmSFlA7qugg5Vlb/J75vhFNc6mfnFGbabzrG9F6
pSq60ris6nnUnlGVeSDU9ZM/TWrwHCGOik/pvPt0eI/OZ1D/Xbx+MKcLs2re
eppf4VVlNO/mRlWSFo/eRvO5e/n7XvNPDK5B/SYl76c63Jn5HdXB1uzvMH/j
UMezkvZ65ds3TSyh+oicTn7z4dbkFyNa4PO2C1HPbRZ71p9RYVP9mr07oZ8v
D8D/fhqpMbOS8lT1b57rh9ox5nM6XjYU/W5eiPwFQ7DuvPAwqWw83Xv9ERdn
kH9VHK1zdu4F1I4v5jOMt88ryOt0mWf+H9dxaEExrRfdY74hLVlB64aU0bRu
qLyYYq5/5nRhZL+ZRvN45FKqR5V4B9WFb1b9VtJpOTM875/XXbeI5v0no+n+
LTLiUHcdWuH+ecmYg9Ru3fVlnvN3YFv6nmDnHSGf1W076vWjqXh+mHLGITxH
TK70XD/uqd1O8SO4T/sii9BOFO7fvtTLaB2xx5bv96z/3Vvofm2m/0r1azTa
Su3I8nGUb8d1pf7VVfeFvn9f/hzVqxm2CucfkYnzXxmBdaXQZ4fKtxK7ZNO4
b29DPnNoBo1XPfsVrZuiqHQ91z9zuhGPZK2g+b+0czrNQ98DnnXvRjafs4r8
kxPoPqUKkqheVfZG/A4Qu8Dz+7vdMTeVju89UEg6rgmtI+roZtRP6jV0P1Qt
l2R73v9FTX+qUzmtlOokuw/u29d2ILV2+/H9o31WyN8BjX1daL0z1uXT+iH3
hKP/QxfR84VMz98QKt98+SA9B6k9F9LvDfaVeG6RKfH02SotCZnvYBWsxe8E
be+k87Hye3zMdc/817DXvRR/UvN2w8cv0rxPj19KdTS5cvCJtCOuvngirT8t
z6U8EdsM69HgnUv/zjgs35x5VMcDqjKpnfGJ80/mPGR68htctwzDMAzDMAzD
MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMMypYt+Zhv3+xetIZd+NpIZW
qeOiR5rn/l0zeRX2Cecux/70t+bsIP9iqLUBccvxufufuZ3eL2AVNcZ7Bl7/
nt7HJVKgSseF9rkR86Npf7L9VAK9B8h8/V7aL2emQGUC4kr73KiXt31C/U8Z
iLxGGbSPz9YqddyaCl9Q/7Pr6bhosY/2EYuzG9N7PMxzoPIiHU+p99zHZxeG
0b5E4/wSRceLJ9D7hKxtUKnjqiDMc/+iaU6k9q1+T9P+QjG24xrK06pidXwS
fG6MB3OofXPGGdhfWT57NbWn1XwVcdkPvqDz39EE57nIpv5s/6NZ1G8t1F6I
uFnUxPM9aco/ZS35l6S9S77IUbRvS10K9aXq+LfwuZGtE+i47B2HfauvRS4h
30ytPXRc+/4Ku5uf3lNlR/lP6H1V1js9aZ+XGt5pcG4g0GJtAuXtXkOqHkVc
pvb8V+4HU9lN8Z6OGzuSGr1i8NlRHXd8QbRrh/315/aEf0w83v+hVem4atvO
M19VdEZ8+kjk70lpoE7cdnxuHuuC/sY9h7xpM3AeWp24oX1BdG+L8Y0ZBN9V
w/DZUR13fG7Mme3R79mS1HogEXlapY47Pjei/krkVd4N3/QByNcqnbj2BV2/
rmdh3OFROF7bA+3VQH1OvNtZnvlGeB3ejxSL62vuvwfXW6vQcaF9bqydNXg/
w5nnYRzlej58ARU67viCrt94xOVD52N8A2vx3iatQsfF+D/JL9uF/Eug5uqj
DdTQcUP73MimSxD/tALveYjPwecEqJWP+B8+d/9PTkF/Nek4vr289niVOi61
79+GuWnBYfp/9co6jP/j/gbqxIX2ubEylyPuzyOV7/2IfK2GjivH585fsw6+
kaXoLyr8e/JrtUcgrrQvaPyZmzC+UXtwfPYFyEvRmoi4pX1B+YV63JsrMe6p
LSnPp1U4ce1zo9psQfuxezGO9q0oz9DqxC3tcyPuzkT7t+YjP7oRxn0bVOq4
pX1B168O/xdzyGpcv5v9uF5ahY47viCaJSFv5FvIm7cLvvm7GsQdX9D5z30J
8Ta6/QGf43pqdeK243Nh/DwC49w6HnnJyzBerVLHhfa5seskzr9sEPx3jG2g
TtzxuTFLj2C9HfYL1p99TQ8fr3I44o7PjVxUjeOFJVjnxh1uqDoutI9hGOaU
+B3n6k7s
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 22->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
FzREKuOm9gmCIJwWPwNkNicp
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 23->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 24->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnA1wFdUVxzeEYghRUqhgiymr4OfoVEi1jtVwZwARUCqMjAqkXiJ2ILXB
QFsUWrJVpoCthK8gCcZc09fwET58MZEPA1yQBKGQBCIYOkFWJGADUuizHcEo
Ned/lzH71gwNdKTT85vJ/Oed/Z9772723L0Pcve6tIkjnoq1LMv+8ifxy584
i2EYhmEYhmEYhmEYhmEYhmEYhmEsIb5b1e8i8p3MGRsuJp9hGIZhGIZhLgsm
vcPrWob5P0Ul19W1Vv+6W2M5zw8M882iqje+05Y6tIdlH2jOE/Pj3wrKd1YN
O9Jau3L4Vdubj6u6u6ovah6oKd3K8wjzv4aoLMhpvm/1439cvznI0GdLYfNx
R0X2NKs7vgHP03W/3k/xfoNPU/yHCX8Luv/1uvsq6PjyrvuoTud9p54+n11+
uFnluFTKt0/nfxxYvxviGqk+j5S9R+NMCR2kfm9uh/7mjDxFn+dETgbl2xlP
UP8i9MXbdPy+/lOo/4SXaRz22Z8cpXZvGdvYlvpVN2zbRHkjxh5vS77IqlpJ
1+HsD1bx/MFcamSvxUV0X2U30PdgrQ9T3VkdUqhenIaFVE/inoc3B9Zv30P0
XFSvz6Q6E1mjqV5Fp/b/IP/JHRS3D2b9PShfjlpFda0yHoOv4APKl1N/gXbe
m476L246FVi/kSU0TrXrHLXvTBlF/bpb4pH35keUp8dmBObrzB70/2J67p10
XHUcg3HsXkrtynu3krrJzweO38P5UQe6bu60RSeonU8/oXq3m57FvJM7NbB/
D7VqKv0e9KDpr1PekE00j4rTZz4gnfXCNq5/5lLjPrAvn+6vn/atofv9uY50
v8mnK45RfGEprZ/tqviyoPtPhet30fHndlF96H7VeN4m3oj6n1FHn90uh93A
57f8Hq0TrPdjUO+jryG1jiSTOuOySFXhmeD66zWklnzPPEP9ORNuxTw0N4J+
w48h//4vAtcPeucAqit34xz4lqegn52Yz6yVvbF+2ZPaav27SSM/ovNPXYx5
5EQRtNQhlfG7G1rLF0sT1lI/WVUfUjvVSWjvcC2pu/P2jVz/zH8L+44VS+g+
C3Wl57kz7xV6LsqB+3e2et9fecMrVP8dPqQ6lpMfpPvVKi/CfCBdWoerbgcr
g9px08Ia9ZVH9721uhR1WzgFz+E/5aEOFx3ZG7h+2HrPyzg+mdbp8vMVqP/5
16L/iqOk4vHbA/PPt5OTTescNbcj1g0Po26df7XD+uGfck9r+fbcZ+nfJ3T6
IDp/edMwjL/TAnw/GXxHzYXUry54BNe7ZxWNRw9tvKA8hrmUqE7jX2vT99X+
o/5MeT+bHqb7t0v3/Atpx07dvZjq9NV566nu0ofS93HnoS1rqI7KcrNbXT8/
334p1dnEs1sor/6vmMec9etoHLELnVbr96XQfPKFH6J8a/AnVM8ivuv6/+Q6
2EndlpE/rYm+97tz+i1py3VUkdylXPcMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzCMhz16NO2bt0sqoAM/JnUHQC0T18YXlT/m3LvkL51IauWs
rsW+QaMmLowvirzMGvSbQ6pjFv2F/m7fqDBxlZsZuA/GDf+c9h861dnkV+HP
6O/2nRKoXYO4VQJfVH5WIu23kU/NoPd8iGXJ9D4AZdQycft3icH7oJL2074m
e8Ih2keoMxpo/4A7EWqNR1xcuz9w/5N+MZ+Oy9ipeD/IbcNp34C4GeqYuJqd
H7z/d0cxjqdPpn1UzjXfp33cytMJiIvtxYHvHxHJRXRcxhyg9xy5nbLfpLwE
qNMOcdUXPj8yfIzeq+Imd6b9izoxXEL+Lkb7IC7XHAt+/0rC0XW4Xn+g/aX6
gXLaN2LfDxVPI251hC8qv/5JGp98oQH5v1UrSKdBrWmIWweeDHyPky6vp3Fa
uTHUnzO/M+0zcWZDxWLE9Qbj+xqc9GOhwPF9TfxyQUyKo32zOu06UhXXs4V6
ccv4/Ki9t8EXMwT77heMgG8hVJm4Y3x+7LJbEF8m0M6jA5Bn1IsLz+fP/3YC
2h8Tj/zKq/DZqDZxaXxRrIxB+3VdoD27I9+oNHHP50cXm+sV+jHyElLguxKq
TVwZnx+nfw9ct3G94X/f7Js+BFVe3Piizv/BbyF+RRJ05ufYx/x7qG3i0vP5
6Qyf9Uv43CE2rsNQqBc/7/PhNp7E/ujIp+i34lwLlSZuHT8ZmO/ceAj5e6/G
+d5dd/qratUiLozPj+hTjn6uVsi7K4R95Ea1idvG50dt/xXyEp9AXmIR/EaF
iWvjixp/5p3Ypz57OPzjXoXPqDRxz3e5Ia8vjtC43tpMKpvebaHn457Ph4gs
o7jYEyZVg3aQaqOOiUvj8+OE3oA/p4zUbr8NfqOOiSvji2JBCP2vNe0v0Wgv
D2qZuDC+KCYVIH58DXRWJfqbWdki7mYWBJ9/YRHiJ1Zj3PdW4Dw89eKez3/+
sRifTDe+65FnGXVMXMcGXz/5di76OVMIf6r5fRkVJu75/LgjTHzgazjvU5ta
qBc/7/Oh3shD/wehVueN6NeoZeKu8fnRTRnID/0G2mMt+jcqTdzzRTErDv59
3XG8dxb8vaCOFzc+P07KBswvIyswz17xGeYho9rEPZ8fUVtCcTttG47LuyNf
VWnino9hGOai+Df6qDNX
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 25->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnAtsFlUahqeyyEWpyHJJi9jBxFIS1mWRlXo/UTeLgoCSTZbuAgcWdi0E
pciaCFaHqkUDwaxSlGuPpSiXimWrlAXcTl1AEMoKKgiFdNpSVqhcSlUQEy99
3zOJ/f+BIKhryPckzZt5z/fNOTOd78xM+5+/++iH7h/bwnEc99uf9t/+tHYE
QRAEQRAEQRAEQRAEQRAEQRAEwXFXZmy//QLyTV7C2gvJFwRBEARBEARBEIT/
J27V6b3yXisIFyfBuIr3z1bfqq6u5qztB9evl/lBEM6T3Jf3NNWPf/3i4Lzq
qMeK2qY893dZdVH5pmZqVZQfZK3/L/JuKD/YpLrP4I+j4ryRz+VH+nddU9Lk
B/2zMX+4KQ8euJB5wKv+7PyOXxB+AvzNHxU3XZ8qp9Nu1MurN/4P1/3gB1G/
zi/+Xhh1/brtE7chbtLuQ6iXCX3rUe9dk6HOEysbsN2zrCEqX9WOPNrkm9Kj
UO/Wl4+h/zZJJ5D3zlZsu7MHHI/K90+mMq+h5hPorDcRb1YvZX9LtmBbTUiI
7N/rmbgZ496/DPXpVraqRn7u5TgeT23D/t2nMyP7D9EDp5RhP9X938a4J6Ri
3ggmlmDeMlPUke9T/95XhY/hPNycsBP5B0uXy/wh/NCY+e1wnzPtindB83MP
47o9OQrXvV8xFs/HbuMf34q6/rR/00eI39IN9eUefYX1vmR8I+rutiOoY5Oc
UB95/b6yA/OMl8L8YEpHxLvJFag3lfEH7nfOY9H1W3IC9eqPzkC8SekF1X37
IN7c04r76b0qsn6D5+a8z7wFaHd3toB6vx7KvMxWmD+cwjuPRc5fj5Q/j/ir
O+K8OalFzJ++AeoP/QAa1E+MPv5wHE+NxPxq1hzHPBRUzcFzj+k/H3nBvBEb
pf6FHwtdtZr3wVV1uN+ouXs+xHX8l014HvBzypdFXn9/egL3O9PmFO6XzqNd
UCdBdg7rtdU01kGHQR9G1k/y8U2s3wLe59sOYx0+1Il5Jd1Z//03Rz5/6yNF
6N/d1ZfPDYv+w3mkeDvng/RP4XvPt4x8f/Bm5qF+zYhpeE8I7urKcb8xi/lf
23EkDj/r87+bcwrzp3/HEM5/f9bMzxvEeadu3Tk9/5vtpTjfwe7J6M+7txPy
vH0bjdS/8GPjFf4W15nXrw7zgVoy/NyeOz/+1wuovxkLP4DmHalEvpuFutDL
K6PnD4uaOfpd1HGPUXwOb/0C73+JY/ZxHtj/dlS+ubR+KtqzD+E9xLtvBJ5X
vOx0zEfquoeRr2tvOevf8UxaHu6vfmFv/D1B9erF95+qZ/D/AdX696+d03k4
tHAH8grKcRx6wG7Me/rKpMj3pzPh1Qzaivivxn6vPEEQBEEQBEEQBEEQBEEQ
BEEQBEEQBEEQBEEQBEEQBEEQBOEiJWMY1/vWbIealqehymroKxsXi5e+B5/3
N/V/hfrzt/L7NqwGh+kbGxeLmZf5HvavC9/j+rtf4nP8ZgdVW1/ZuFj8fbMY
N/cyqN+vags+/59OdebRD2xc3PjLemLdgWr84h3kb+yH9UjOJqp3gr7xGRfX
/9P7EacPXM/1eXmlXI80h+rUWt/GxaKqZ2/A+PZuQHxQs7iM6w6ooa8CxsWi
U2agXS97yUdeuz7rMN7LqWopfb/bjMj1E/rkOLYPq1rP9VNDSpGXT/Uy6Aef
My7u+E+koR+veNtqaGonridNo/qv0VcNjIsjN2cN8paMehPtyZf+E+PoQnUK
6Ac5jIvrf1pbnCdz1WnEeTcvxTp2fRNVdaTvPs64uHw3G/s1X2ju/29rsd7F
jKL6p+jrlOzo/gfevQjt9cMxXvXvxUux/RbVPUxfD2bcz47JXbHeVg//DdTr
0FxD33+YcXGMT2X78d7UpHRoYNWE/rjUyHx/6rWMf7UnVD3wq2bqWz+Mi8sv
5bi8ocnU+7tz/bBVY31nzRnGX92W/U1gnLmmG8djVVvftXGxBMO4XtkUtud+
JnXmeK2GvrFxcePvwe85MG/wOM1ExgdWXes7aUmR+eYA+1EVacwb34Pjthr6
Tl3nyHw9vQ7rFb2iSq6z/KQF860q64dxsagdbTjOzC/Rrh451kzNA/S1jYvL
b9HI9ZIv2fbKfQ3f1SD0bVwswYoR9Ft24fiLy7m/16na+n4YF5vfeQzXm376
Gdd57u3VXBvpO13GRK4fNyO53ttfUM71nmYxt0O1vmfjfm74l7yOdfqmtgzq
z9zZTHXo27hYdEDfc9axPaEC6lr1rW+C6Hxv6jJ+T8DWFdBgwCrmD7Rqfd/G
xeWnFdKvyec4kpZzf1b9avp+GBfL/IXst4TtXlEJ+7XqWz+YtzAyP8jluLx8
jtPtvJZxVo31w7hY/K6GeZMKGJdZzDyrKvRtXCxuFo9PT2e7dxXPs7LqWV/b
uLj8tDz2d8Vc5q0s4nishn4YF98/z4sa8iLbr7O/T6uu9XXWGc5f2pM8vn7T
ON4t9nxbVen0w7hYVAF9/a5m/qB/MM9q6LsF0flmdiJ/z890YD+HhzRT91n6
YVws+tok5udfyfhdk5upY+iHcYIgCBfENygBJzw=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 26->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnA9MVdcdx++qYicFHy1aK7W5pAY3u7YbbTJHZZ7+jTbKDP5rG1tvao3O
yuzUSXQVj5txjdCJ07UKm17FKSJYUAoVqVxR/Id/cQpVhIsFq4z6f9QmhUx+
33OTvvtuXlvsolt+n4R8837n+zvnvPvO79z3yDsv+vXpiZO6aJqm3/zz3fy7
W2MYhmEYhmEYhmEYhmEYhmEYhmEYTZbmnRhyuyfBMAzDMAzDMAzDMLcJ8UXF
Sf5czDD/n5i/ubQ7aH23pZ/l+meY4Fj7pmd1pk5kjG9mR56ZGdmp/z/Lbq3V
HXlGzLZqr3xzZto/g/Wrx506Q+39Fp7pzPhWw5lSmr9vzy29T5BpDcd4n2Hu
VMzpq7eX3VQ7NHFbxzq1EvrWdKhIij3SoTLy/TfKvPKW1O2k9hjf0Q61n80s
pnX+XNRx0j5XP6P4sU8ueq1/e+BjFzri+tyRVTReczjVuRlbUk/x9Tcoz7oQ
dckr31pSQfnmZ3/9hNonPEFqJyQ2kY5YcJniE+d75svSHVup//iKpK+3mxm1
+2n8+jnUj7VgmOf8vwl9UVg59ZNVdf5W6t+On5rvdf0Z5law0yq2UP2m5lHd
COPsFbrfdp9EakalUX3Jgq6HPNfv4Rt1FH/q+HXSw9FXqV6eXHiN8q6HUf2Z
yQmXPe/fy+NQl2PzaTxtukb54sE19NgOP468mLc9843rp6mujIRyajf6b0de
ho/UOhJHcbH3Pc98EXeJ7sv27odbaL4+7BfW66/QvERRPK5H27Sg9W+uXtxA
/YSl2uTvt4X6M+4bQ/3oA4Z5jh9ARu0+8i2/h/ZT7WcV52g+p3qW8/sH5vvG
ntG2idZX5Fq6/9rZibSO5dzxdN824zLo87H97+xdnvU3t4DeFxvDmqg+5MJc
7BvnUmi926Nm0GP5pzkNnvWXlEnjyo1TqE7MluXYL1YMRT0v/TH2hcwIz/u3
Nv8PByh+LAvjDRiHek/sjvof/w/MJ/3X3vV/oWEH6vWjz6m9tAD9JP8K8/hq
L3Rlz6D1K058+SmN12cgjadP1vC89z2F/SR3RmOwfDkvJ4/8b46g/UNrD8X1
f+FRqn+5fv52rn/mv4VekEf7gMwVlbTuhu/GftC+6WDQdVuflUv10+tKFe6T
qXQ/1iu++hflX1pE7x+s0DGe/egpVRupfdll7DOZ/VAvXVqo3mXSk1SX4v7w
A56fH9oGF9F49/yN6k9L6I96fSwE/djz6D5svZH6cbDnoQ+KP015g0fSuEb6
W8j//dBm0iHdKoLWX0qsSfOvnk11bj1dhX2k51p6/trwIs/5uzFjr6yh/F0T
aD7aun2VXPfM/xqi/tMVnVm3xrjmDKr3Px/9kPLDn/tW/Rg5s96henut/APK
33ia/i9h9I7e+l3mIVcdpPuwSN5L+4UVsji3M8/D/mNZDtctwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAM831j9HiBztvqdUtJxcs1pPZLNX5xU/kC
8ttMOn9v/u5FUnvWDjr3K5TasxB3fG7M2bF0zk32Tia13r+XzgnoSk0Vl8rn
xrKeIZ+8sZ++Jy8aL+LcbhNUfIm4tvMZ7/MH29rJp0W07qXxG+fsocdNUN2H
uCxRPje/WLUH3zseSOcDRPwP6JyU+CXUeBVxOQi+gOe/TtDvk1jFf8E54Ys1
ZZTfDHXiVpbw/B0T8Xw0tZtLR1s0j8IoOidkKZXpiJvPRnueH7SS6mkceeUn
9DsHIi2Uzm+Ld6HWVcStafC5MVJ6ldC8nziM72tXL8J50hqo/lPEzbfhC5j/
S5epXbQcIL8YOSqf8kdAxXkVHwNfwPyLY+i8ut46cTP5jt63gfQw1DyHuF4I
XwCPt+djva2k8y/6ka70/XN7F9R6D3H5KHxuZFlMGo1XV5qN1+nUetK1UK1W
xXfBd6ehpwzCedvPR5Nq+lg/deL2PPjcGI394FsQhfbmB/xUV3Fd+dxYg/tQ
3MzsiXFSQ/1UqrjjC8g/FYP8uEfg//lAjKfUVnHtdIx3/rQH4TtxP/IKe+P5
KNVPIu743MhxPShuTA6DX1zDuT+lTlwonxt9vA/z7xoJ7RWB8ZQKFTeUz40d
ch7jrW/Becc1uO5Cqaniojt8bswpbTgvPaEvxtnfivOSSp244wuYf+MX6P/N
drTXVfmpPRVxQ/kCeKUS+Q+chK+kyV9V3H650jPfuLgC+bXLcB2e/xiPlRoq
LpTPjSicg/6HbIKvfD/Obyo1VdxQPjfyBM6Zyqgp8JdG4HVQ6sQN5bvTkG0F
dE5fj7BwXr/XUVJDqe1TceVzY8Zvhu+1UvRzXuUr1ScgbitfwPg7tyE+GuOI
7EqMp9Qehbjl+FyInELE39pCahaVYHylTlw6Pje/zUdeNeYnw0v81FBxxxfA
49mY55IczL/2Iz914rryBcw/Gf2Lh9C/sVpdR6WGE0/2vn725g2Y7+hN8N2N
eetKTRXXP9jgma+HrcP4IWg3sovV61DsF3d8ATz9d4wzfDV8E/PQj1Kp4kL5
3FjJS+HX0/E6/3Aj+ukBNZ248rmRP5qJ9sXzcL2LZqt1MNsvbiifG5F0F+a5
NRLtB/3VVnHHF5Df9xD2/daz2J+7hVz7ugoVd3wMwzC3xH8AScQ8Gg==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 27->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
mCviTwl9PHk=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 28->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnAtQFVUYx1eltMgEBzE186BWZpkhPbTxcVQwsZSpzMZK3XR8lJU9bUqc
todGamokWWPmpl4yRQQVTVFYEElN84GWj4pFFElJRUwH06b8/ucy3r3btbBG
Zvp+M8x/7rf/b885y35nD3f2EDF0zEPD62iaJv78Cfnzp57GMAzDMAzDMAzD
MAzDMAzDMAzDMAzDMAzDMAzDMAzzv0aszf622+XuBMMwlwW567kDXP8MUzMx
j9y8PFB9mnmDvgp03L5h5maub4a5RGbHvpxdjTRreosVVH9j0w+61aH8dMGO
gPVZPKKAjteuf8jNp88YtS1QvnVtE2rXqt/Dtf2LIWLi1p3Ps5MLfrykeSRp
2laeh5iajmUvKaL7fXJYyXnVExNIjevW5bvdv/ZL+h6KtyykOjXbReVR/tcL
d9PnsMxSqr/GOUdd7/+i+CMUr+j1/XmVz0XupPZeCqV1udW2cTn1Iy/ZNd/I
iqF8u34W5WuPvEj9kF2DqN7l6LHH6TwdU8pc54/HK5bR+fvP81w4v8lXm+bS
eWtNwnman3LNvxhWwlep5/NEyb2l1cnXK0tXUz9SG31ZnfmXYQJS/8kPqT6K
Xyymeu3fBfUydCapHf8oqVyR4/qcFRNScF+vPXOS7vP8WKpX7dofKU+0b0mf
5Zvhx93yrTGNqK7tkvtPULsTQsmnPzUJ7S5bd4yO/37INd945yjNV+KxTuWo
8xm/UN4StK8XHkE/Vs9yzTcXpNFzWQ7efJiOz72f1hn2yN7kN9K7QSdluOZX
nee7jd9R3rHkQrqOr3WkedP82MQ4Bhw/9nfqX0RFZlK/u0+h623tPUvn0XZ1
zeP1A/NvY+TmzKP7e6VOz2vjhQP0vDNifqPnnVi6gp7DRpvKNa71G3V4F/na
xOM+32Gh3m/bTfe7Nf5uqmtb7+u6/rbyU7fT8b4dUF8RO2k+MLYkIT8I84Do
09W1fsyzFQV4fjdH+xlN4X/+TfKLcSWksvaDrvkyrJ5JdTq1lNYRcjzmDW3k
SeqHOaoN5pXKOoHr97Zo1P/0geQ3D6zFPLqvNtYvzdsWBcx/RiRTXnxbup5G
bOufyb8+B3/3pB/M5vpn/iv09a3m031Xpzt9X2acG/4N3W8hnwT8/kyGeDxU
Z0llVIdGj4j95E9rR+sCMyTcpjpYUbDBdf19tFca5TdbjPX/LW/QvCPzo1E/
N2+i+99KKs11rd/o2aib9Ld/ovO0Won1wrkWyE+Zhef62ZiA3xPKKyfQ89au
TMA8sGlkOdY/8yhfDosNmC9em29Re1NuovFbp/Ox/rh6MK2rRK1FruN3Ysya
S983yJ5F9HeQDDsY8PsNhqmJGFZEUnXuWzvo3GSaR/YvwbqkYmPiPzrPr/XQ
7p59NC8YNxz9+J/kiwLrXarbVf0+p/rblj+xOuOQWU8kcN0yDMMwDMMwDMMw
DMMwDMMwDMMwDMMwDMMwDMMw/xVm4RS8tz7yDKkV3YDewzWU6iqu2fA5EUm1
6D1VMy6e3v8Vj9q0n9cYALX7Ia4rnxNr2Gr6/zvypnbYhxPZdBOd7w6lKq4r
nx/vTSSfYZ/eSMdD69J7tqIhVC9CXCqfE3v3EPJZqxrS/kYrIRH7F5VqKi73
DnF9f9dY1Ho9tXPwGXpvV8RG5ZCvD9QqRlwqn9/1uyYrF/sbivH+8FPP0j4L
MQJqqrgZnOX6/rMxegqOf3QaeZkDV2I8UH0G4sbT8Pm1XxCP4/O70fvNRkgQ
7Yc0QqGWB3F7e7z7/o9BObS/W57KTqfjd3+UQtoJKioQN56Az4n+vkHH7TeG
LiJfYcQCGs9eqPWqik+Cz4mddsVSyg/LmEPtjSun9z/NsVBxZjnFzcXw+fV/
fZiJ97TTprr2ry7iMhe+v0KWvTKT+tlwzFzyh0KtEsRrKub1N9L+HFNvSWp4
8Fl6fOO68jkRxfdRXNQbCP+6/j6qqbimfE5kTA+cf1UfqKe3r6q41+fE6H0r
9hcNvQv6cHO0r1QbhrilfH7tZ7SHv0Mk2nvodoxHaVVc+ZxYrRuj3XSBdjrf
A18XqFiq4srn1/9PwnH+vDsxzlD0W/eqint9TswzVyGvSwv4FgSjP18E+8S9
Pj+CD9M+B6N7JaloEnHiQq2KK5/f+DO3Yr9Xxe/wTSuBT6lQca/PL3/QCcS3
hqG/WWqcSnUVt70+ByJuL/ZbzQhBv4954FNqq7hUPj86T0T/k67Bdfo1ufxC
1VTc6/NrP2oAxtfoPVzHVInzKbVVXFe+Gkf0rgrq95w9Faijkz5aFe8JnxPx
w26Kiw6F8L1+BJ/HQQ0V15XPiT2+CP6+xaSyVRn8Su0HELeUz4kMR79szz7k
ndvvo964Ee7ef7vnTviXFsDXzPZRqeJen1/72XkUN4M3wH8a7ehKq+LK59d+
gzw1/nz4I9XvQ6mm4iLEPd8qy0R7URau0ws7fNQb137JdM3XgtZifJ2yoZvR
ruVVFff6nIjEFMQXZyB/9jb097NtPvEqn3P8TVMx7luX4Dqt2Y7xK7VUXG+W
6j7+hYnI/2AW8hZuwXmUGipuKp8TvUMcfPOHo70WHl9Vcal8TuSiThjfiBho
3FvwKZUq7vUxDMNcEn8AFhotRQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 29->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztm31wFtUVxncAi9VCU1qnw0dlA1PUUgZBBrAiWTLQEpEWI2P9AlZKE8b+
oUViik7xtk1VMgI11tpGhZUAQijmQwgUEtmQgq1MICYBQyDmAgkxAoYPIdFS
p+Z57jJm350opRZGz28m88ye+5y9u8s9d3df9sbPfCD5510ty7I//ov7+O9y
SxAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAE4UuNyknennCxD0IQhIuC
atzcKPUvCF9Qbij2pb4F4eKiF++qiapDtXjJjs7qU9/UY1t7u66s2B3lc19M
+kdn+X5V+YH2dnfs1Mj8T6V8RRHyrsgvvaB5JHlOlsxDwqWKbhn4cvv4tL//
tUrUy2/X72lXb+hL+zBuq8c9uSUqcXjfKuT1yClrb/eqqqo+6VN138B+1PDH
TkaNf91jzmHU99evfrPD/hMn/b097hTORrv1kz1Ho/K9hMyGyP3Gp72O/Kxv
NkObEyLzPw2n3/Rq5FfmNF1I/bojB2mpf+FSw97qrWmvO+fhFNwn7eZ+R1D3
+we1YNw3jDuB+n3m4N7I+3c35xjqN7cN9eH1/zb8/rKVyPeqF1GzMyLr351h
HYf/2gzsR/00i9s9t3G/Q2tYt1M+ei8q3y9bfAjHuaAU7c7m/pgvnBFHkecN
GYW4qmiNzLeuWr4B/hvrazHvtSzEvKObFmJeceY9gv34U3sd77R+x89I+WS7
l3oN5h932gEcn7930bH/pv79pJH7kbe+vEjmD+F/jX5nejHq4/CQg6j/0r2o
Vz26N8a7e/9c1l9iz7LI8edUIc+/fwbq2/tTEtR//jLMA/b8uPehM7Ii608V
3Iv6UMldmFf9LOeLCcWs/419OW+kLD8R2X/1rzEvqRsH4Hh1S1eoM/tWzgc/
3IE8ndw9Ml/np65CfyNL6L/tDvb7XB6vw8CR6N8ZY0X3b7BXX4nnHP8qB/tx
H9rI/O3ZzM99uNPnB7Xuznk4jzMf8H1oXxt+l3RL0nD91azX3pD6Fz4v9B/n
78T43f9L3AftRU+hrvSck29j3BU//bOo8WfXjMX/n3kP7MJztl9zAPVn7zyF
8W/tuI/bb/0h8v3fTn0c84qfMYvPHXelcd4ZPh915L+3ms8hE7L2Rd7/a6/A
e7leMgB166bHsd+CPOTr0oFmPmiNzA/wSorx/GNd9z3OFz0P8jjG3A718lZE
Pv8EqCdL8J7grvV5vtP5vOCOnsbnBseNPP+Y/VRnoh/7KysaOH/U8/mmsPuz
Uv/C540/bNITGLeDPlx2PuNN7SpUqNehk/Nxv/qz/SbGbcGAnM+yH29B/9XI
r30edWTV1NShDrJ0BbZfnbiks/34h17H7xdu8b9wH3YrFN63/ReW4vcJp9uy
lZ3W76hjL8HfdTf81ncHYL5wH9vH+/qPS9Z8pvo927AR/t6p+L3RHbynBHkl
9UvP63p2cZCnEm9+QepeEARBEARBEARBEARBEARBEARBEARBEARBEARBCKNy
buf3clX5ULttFVS3Uv1KxrXxhXGSnsb3cl7GB1g3qH4zGWoZ1b9j3DW+MPqt
vHL0s20vVaVjPa9j1NrOuDK+MN7v1T/hb34Ffq/uPnyP7L9Ntd9lXBtfzPnP
dbBOR62pwLoff0HrVuRlUnUu4zqNvjD+0W8xb0Mf+J3kwVugU6l2EeOe8YVx
Xy7E98tuShz81qlhf4O2UFUq43plYeT6Y3t8PNrV7KWb0J597zp+f0n1Uxj3
E+kL41z3BNrt3Ob10MbjeejvMFWtYVxfQ18Y79/16MeO/9EraE+bi+8s/V8Z
7cO4d5a+mPNvTEC7zh6zAtetrIbrMUqp7jOM+/X0xRx/xl/wvan3fhq///Sn
wG+/RnXfMXFFXxjVWpeL/nvdAp8/77K12F+6URO3T9MXc/x/rcB3sur6F2du
adekokxsT6FaP2DcLag4r+9p/1+oX4zA+hTV7yaofVcCVN+Z0CHuGF9M/sSJ
XLfzFNW9ZwLzjJ6LG18Mpbex30dnsZ+6O+g3qh9hPPCFcdbewvzM6cwfNo15
10/rEHeNL4ydfi3zVg7h+qWSRPqNBvHAF5OfOoz9bTT+7BuYbzSIB76Y/KWM
26PG87rljuZxGHVMPPDF0Nad13nTd9hfGf/dXKPKxANfGD/hCNc7FFzOfr7a
o4MGcc/4wrjxXLflTRpMXUdVgZq4Mr4Ylp3getFuV9M3+yy2XaPaxG3jC6Mb
ahlPv5LXa2w3bht1TNxqrI0+/nFx7L87/W6fKcwzGsSV8cXkN73L9R1949m+
uzfXvRnVfRh3At8lhrtZn8Jx3toIVXd/1EGtyYw7xhfDiCbE7ZPHmVdkYb2f
Z1SZuG18YZyFJ9n//DNQX3dBnjbqmXjgC+M9eoT5d7ewn6YP6TMaxLXxxbCB
52eXHWb7zW30j6FaJu4YX8z1O13H484z1+/MaW4b1SYe+MLYLZVsf7CG/ehm
Ho/Rc3HjC6NX7WR8ZjX77dfcQYP4OV+YuFK299/BfpYfYv9GfRMPfDFsMvGs
cvofauD2XKpt4vam6Hxnz1bGe73B/nN4vspoEPcDXwhvPNu1z/P0Xt3N62bU
N3FlfGH8/Ez227aa+Y9XdlAVxI0vhv0P0jduHvdzz3M8X6OeiQc+QRCEC+I/
EnpUrQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 30->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztnA1wVcUVx2+QRCQOtEREAeWqtGVAJo58SITA0o4EagoIFkQkrIoSiiZp
ASla6lqIxYEWAYdOQNtF4ggYhBgiAaO5iJFE5CtRSiTRG/nIBxEJkaIGUPI/
ex1y3+2rE+3IjOc3k/nPO/d/dvc+9ux7980u192XOuaBSyzLss///eT8XxuL
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRjmx03RpNeH/NBjYBjmB0G+
n1nF9c8wDMMwwbgPrUtryeekO2k+8vTE4btaki9f7bu0Kc+ecklOi/KXZpY1
5cl3a1bx5zzDBOOc6ZdTcF7VvJgtqJd2jUWol9b/3InXMaedgqBEsWF5s7q6
Nqmmme+9Dr9uuu5Ezq4Nqj9566zKcHWpC+rcputuVUVdkM8Ryz4Mly+Xu4ea
rov57Y61aP3Yc/MO3H/DobDj/J/tdOjwHq8/zMWKmPH0bszPEcWoU0eX12De
rxxU36QqosgNmr9qZSH8MqrTCdRpSjzqUV556DDaSf7yKLR448nA+d92CvpR
nRpRH/KKv1KdRR5Dvlg3gtpNn/5pUL5tV8JvP/lmCfr5xWPrkZe/+WPEx/4U
+eLcTcfD1Z/62aWLL7zuZjxTiLx+k9GONXR72Pz/2q6dnYf7G5RU3aL6v6zr
NuQPnLs+cP1lmO+Acp5ch3kZPwD1Zm+9F/Wu59yBurFmFmHeu435e4Lmr/u4
PoL5qac2QLMeQr6V3ht5uqdAO85z6YH17ybejrpWd9H6YB8d+An8d/ZFnv2v
I/S6OCWw/kVyAtYlGdGK8ucfwPcE+8HFtG40xFL9d3sluH5L9Ca037VbLdXZ
M7gfPepZtKPdFBpHx7T6sPVrr9sI/9/lAWibf9P6F7GF7i+1+4lvVf/1Mq2p
zuWsilKMe2pvrI/ODWff5u8PzPeNuEFlYp7dan0AvfwIfg+3T/TG57YsaaDf
xwuy9gXW37m2mKf65SrUhzu4B9TOXEJ1MyUb9Ws/3KMmKF+v3Ezryi0SdSIO
rIdfPzqa6nXtpVS/tUsCf6fXuzKK0X5jOnwqmvKlnEP1u/8TWjeGLgl8ftDX
ProMvlVHUPfOX9agHbnwNqrbIfFU91/0DHx+8bAjy/G9x47ujf7VnPbH6XvJ
frSnb5wQ9vlBvDQ4F/2W30PfZ0ZV4n5lThnGrV68Jo/rn/l/oRIzXkCdReXi
uV+dnv4u5lvc3zaFfT7/88cvYp7GzUUd25t6Yf46E1PxfVe+Vojf39z9vYLb
ae1moD7a52IdceMW0nPHa1W0HnzVmeq+T+vtgc8fSV1XIP7ZCnz/F1v24Tlf
ZC5Cvr29lJ4/xty9LWz9HovA7xzuqkj07+ZF0zpwYgfqTxzt/Fa4fEc0vAT/
0vsqoDvHYR0QsdX4/cEadLb429Sve+rgGvS/4rn9GHdJ1l6ue+ZHx+cHZ7To
eftku5nIqz0+C3VYuXt2S9qxyyP+yHXHMAzDMAzDMAzDMAzDMAzDMAzDMAzD
MAzDXOyoJ67GPjNljab9Zsfi6LVR96tRpMbnR/wpDvv2rNmr6PxNtzLsE1ZG
5SMU157Pn5/xBvYZ2oXroWJPFM7bWUaFiVvG58dJWYj9iuKqB+mcXnYR9uk5
r5DqqykuUskXwsiR2FfvnH4e+wvl8/fQPsHVpLYXHzUycP+9HNMJ19091Q7a
uaxjPnzRpGo3xW3j86P2vYnrKn4D/E7b2FehUaTuYIoL4wu5/6cS8P8X6a3d
sT/YFb/NxniHkDp5FFcLEgL/nyO3az6ui81rcL5aWC/QeZBWpCKP4qpLfuD+
Y1vG5ND71xN+1bc7zlmrAaSi3sQnxwSe3xYDNlM/4+PJn9Cf9qMPI9WjKW71
IV/I/S9KwXkvtXou9q+q/rGrkdeX1PoHxZ108vmRj+SuRX9JVWvJl/cyXs8j
lZMoLmeSL2T8Q9MwPufkTfcXNPWv738c49hIap2luDUs7aI8fy7GjsC5GTlx
EunGJDpHY1SZuGV8fuzfD0PcveYOylvyq2bqxS3jC2HtbxB3Jo6jvJF3QoVR
ZeKeL6T/fdSPkzweqo8mU75RbeLC+EKoTqR4F0n9pYyl98GoF7c9n5/a/pR3
WwL1s3gg5Rv14k5N/8B8ee46ep+S+1DejOGUZ9SaRnFhfCH5O6Oo/enksybH
Un4SqTZxYXx+VN71dH8/p/GJrfE0DqNeXBpfSP77dO5KHoohf11zFSaujS8k
f/iXdN5sXCT5E0vx2rm9tFlcG58fNzqLzpv0S6frMwrpHMofSKUXvzwrMF9W
TaDzrbm5lJe1gfo3Kk1cGZ8fPXkvnet6qzuui11dmqm7neLK811kyE8PN9Dn
x3Go9Z9Wn12orokL4/OjxtN1deYM1C3vQPkVpI6JO8bnx+3YhnwPx5BW96B+
jWoT93x+7LkRiKtt0VB7gU0+o9LEpfH50cvO0fgrWlM7JZ3Jb1SYuGt8Ifmd
6hGXB819DrqC+okn1Sbu+ULu/0AdxU99Tu/fcnrflFEvbpfVBfd/uILiNl3X
jVE0XqPfxD2fn3fK6P5VFV3fau53C6n04sbnR+4oofyllUbN+2BUe3Hj82Mn
fkD3/ctq8k/7kPKNfhM3Pj+O/ojiaTvMPM6mfw+jTqqJez5//xMW0PhKnyXf
qbxm6pi45/Mjek2j+sj5HfW7YBG9NipN3PMxDMN8J74GYMA2wg==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 31->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmwlsFdUax0dULNTtgQJFUw+LVkRcoii1lh6pNPhSHrUoLkCcgpIHiixP
rVbUEwxaRECgCEQIU5a2bA9tG0EoZXBhKW7UAmV5ZaiUvgrlYhcsgqD3+58x
6b3DTUGjRL9fQv653/y/swzznTm3ObfDkFHJT11oGIb45d+Vv/wLMxiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYf6WmFOSDsX92YNgGIZhmL8Y
asHt49b7NfZA6p/xnrXqdn3O73eGOUtKL337d62bFyZVerUnw3fuDNWPqm3Y
7r8uH9zsmS+WTN0eMv9J337kZ5afy3zksRGf+fPMqKs3/5b7YTafvprXIeZ8
xZzyQ6b/PW1esGgvPafhGY5fnZdvraHnvzjvG8/623dFBcXnrM2mOotYW+Rv
x1gTa/tVHije7Y+rinuOeOWbpu9bz7qISV7oj9szO1dR/20qvvP07R1Y6hW3
bmm5iMZTlEZ55ivGOX3PVw9k0/7BHtTBc/1pMqn/cbj+mfMNM23BHP9zKSYv
oToWb1b76DldLY+SDv0X1b+1bJBnncmU9MNU3/EdKc8e05X8culmypfHb6im
drtE13rlq3XvUJ783+dUn+ayzsgbUUhq35BI+Wrs2KNe+faoHKxX92b+n/JK
smg9caJvozz53y1Yd4aW+zzXn2s2LvevU2L3V1/T9V4jsun7SkQ+1at98B6a
n1P+2vdnU78q/a6lNJ/Bx8pxHy707P9MmL7aXBrHG/1pHDLKt+Fc1g/x+E/v
8LrDnAlrVsk6qu/h46n+nX4JVGfqinrUTcnHVEdGVlvP9789eyT21Z+8jXp9
3yE1N6Sgjru9SCoiCz3f//ZgC/W6fyLVl8qJR94Wg/q1D8dS3CpYV+35HE9L
w/4/LJbyrHpB/agT3aje1PPllC8LT3n3/0Xmcsq7oP1B8i9RlGc9XIpxzLWw
/rWK8Vx/XGS3Z3ZRfupXWC/Gb6T+RJnCelgzKPT+4d/pT9O+KS6e1llRVVyB
8UzEutFjQ9HZ1LFM6jWJ8oednIX9W9dMXgeYM2FdG0Z1ZG+/g76Pi0W7dtDn
z9Los3Pbes/3iBNeSd+PhWhD718nrRnqJ6qKVCwcg/1Bt5YlnvmHEgqon3Er
KF9VpaNuazpQnhnfnz6bheYOz33+gGrKNzuVUf06l83Dez95KNWfnfQj8rd1
2R3q+Vdt76W/ExhbTKxXb5xA3eUVYn8SNm5vqHxrTwvaP8h/toV/aRaNw2mG
/u23WoXON8f1pnE+t4vasVqbNB9rjMS+7PWSRedSv/K+4TO57pmmYt/RZzI9
rwfmrjib58YetGcCPb81o/Lp+X/6BO0r5DMpTWsnqvsyqvvIgZ/iPboC3zfK
I9dQu5fPmB+qHfF4F9pvG7XDaR0TLz26j/I2DSimeF351JD1X3x/FvW781mb
9GQ2vc+Nwwkfk16yMOTf76wJ9X3ovtV+s5Z8j077hOavjtN8xITOS5tyH+SM
eWgn6QlqRzbLXMz1yzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMw5wJOWMrzs8m
taDzdfLm3fTZ6QqV/RA3MrZ6/s5ODTxG52ucPg+RisqR2yhPq6njjvYF5U85
vZV8nx4gteYfp/OEtlal464vKH9x/ibKO2VupP5nv0jnbtQcqDRSKC6z4AtE
ZKSin75T6ZyPPFW9nsZ7GmrouJoJX9D9K03E9YYd5Ld3jPyIxlEKNY4hbmhf
IGZ4Ma53WkDnnFTtgjyajw/q6LjVUvsCxz97Jc4bZfT6kHRNMZ23UqugxnTE
5SztC8A+eNMqzG/9B6RyIp03UvdBzXcRlxXwBWK1mU7XjXYxdH7KGvLAbLoP
I7T+A3HD9QU1sBrnrhomk9+IiZtH846G2ocQl+/BFzT/yJUUF/k1OB+2vPu7
5M+BGu/peNuV3vnJ0XRdfBRN/cnrk3Jo3h2hchXiIjHa8/yZWFYwtynns1R+
03x/NM5FvfH7vgEP4Zz6Xf1IhVY3bmlfIOaNj8Gf+SyubxrbSN241L4g7oyj
uCP64nrY4EZq6bjQvkCs0bhuzzehkRivclXHlfYFzT+2P/K+H45+6ofAp9XS
caNnf+/+K2MQvzsZ7Ux9CvPR6sZN1xeAKozE+Bp64j7F98BnrZaOm9oXNP72
ndBfWSLylzyIceRAxT7EXV8gZlQE+ku4H76e3eGLg1o6LrUvEDv3YviuvQm+
46fxOymtSscd7QtELt+H32ccaY5xrPAhX6sbN7UvaP4f1MCXoue/zsC8tSod
N3JrPPNl5WLkV1eS2oNbYD5ahY6b2heIdTAJeX3fwnynvdpILTeufecdPY/S
7/JU3Qn8Pu+aq+po3lptHXdij3r+fs8edhLX81rU4TnuSGpqFfmIO9oXlP/6
lfB9eR2pHBaDfK1Sx11fIM7a1shvfT2p9UgcxqHVjYuC1p759o/h6KddBPJ7
3Il8rYaOO9oX1H/2RfB3bIVxRtwKX3uoreOuLxA5uoHui9yq71O7GzEOrULH
lfYFYqbW4/6HN8d9S8d8ldZf49oXxOgqiotJaN8qwXyVVkPHXV8Qq/cjr8aH
dmbg/0lqdeNS+4LuX24Zxe28Q9DeR+BLONIoLvPKvPvfVoT5p+TCd7oA49Bq
67jzdZFnvhRDcH3PKOTf/mYjFXt1XPuCmJWI6xl90d/743E/tCodd30MwzC/
iZ8ByuA5nQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 32->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
Tr5Tu2hbR+no2scwDPOP+B0OMSSL
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 33->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
hkTqJy6DjofV4njYJwiC8KP4N7w7Hdc=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 34->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
/7/uRPPPDQSpreOuj2EYpl38D9vjOnU=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 35->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
hrkg/gfFrDLn
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 36->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
YRimU/wXzCc7Iw==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 37->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztm39wVNUVxx8mRMjSEhgQSoDclKlIlNLSRFGkeYokpUAFUgOKpE+C2viL
TDu0JcH2SQwGEi1CqGUqeiFF5UewkQCGYnIRSQlIEBEVNfogCUSIUYxEtPwo
+Z77nO7uY4eETs0f5zOT+c5+7znvnt28c/dm7yZ2xqzJd4cZhiHO/0Sd/+li
MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDdGhk
XkJ54rddBMMwIbGeuPaNdvXptIJnW/OcT2+u9Mo3J9S/Fuq6YtgNGDeNmh28
TjBMG3kw91X0z9mJb3r2z5grvX2N3TKlDPlX5dR5xTkjdq0LlW/Omb+tdVzV
ZX7anv5V3cY6rXn2s+mHv9X+f7FqC68/TEfDaqx6teK8yq9GftB6f8ovd+1G
v8RX7EXfjet5FP1b8LNPPN9/D8u3L+a+VtcnHvWKswqyPP1v8m5cXN06btW0
HGtP/zi/LsL6JEcMOnIp/afGxVzS+mH/o5b3H0yHQ9U8g/4yhy5Gf4vC2cfR
/0/+5GPsu1Mq4dtlTZ95vn/v2dSI/lq4/F/I/90Q9JvZtAL9pl6JQ9+avZRn
vt3c87iXLz6sxvXMSRsaMN47pbFd7/9D8g6ijrwx7Vo/XMzpnRouqf+zwg9y
/zMdDXlq2XLcl++8if60bu5/An23Jxd972TkwxedvnI8+3dxA/rc+uGMz7FO
PBONeGfsoSb4h5Kxb3fmzzvhuX+oe4P25zXdMC6s57HuiDt9yHdGjUXfW5kj
PdcPI7Uc+2p10+P7EXfN7k3In/cdvF/Lx/riOtKM8Jz/gq/LsFu3t+6LHNHt
I+Sde61N+S7WmmzUYaYmhNznXDC/uArrqdifvI/XD+Z/jVl77p/oswlGPfpk
cyz1/YI+6Dc1fDTuW7H3t3u97j/11m3090Lvy2j9GB+GfrfeK6XHg06RTkvy
vv/XRGJfLO+bgP6yX3gB/W77NlDfbomE7/Rb6Pn3vzrZfQnqi74c64bZWIf9
hGqZh+chb92J65hR5SE/P5Bl3dBn9m0pVOfrD9G+5P7PaD1LrfRefzR25Khl
+DvqT6nYZ6iDGbSfGngf1bHkjpD7DzOx+47WfKNH/iPQBB/Fb6fXU8UVtmn/
Yd/7xDZcZ1/ffbT/umsjrx/MhXCO7X0J93/aJvSzdXjqHtw34Y96fu7uIgfv
ofWj33B8fqDsdXSfxrwPlWmf4P1TnL52u+f+YVgJ7ksn+x7sI0TCGfSrkb6K
3v9nH8e6ZK/4OOT5gXzynQ8x/wNHaP0Jy6D16++L6Hqxv68J+TlDTjrqU1Me
pXkf6Iz9jFEzC9cxv+5XH7J/p5dUoc7mK7FumPufRh0iJ4rWj/DNF/f5QfV1
6Fdr5iCsg2Yx/T0m4heFrD+onrCf0rlI9kPV3PfM/xu7KD6nPfeducy3GPd/
7IHVWD/KMx+7mOvYJbcgz7lh7Hrknyx6Bf28ZMjLeLw24p6Q/Ttx6EKMnx5S
irwz/6Zzj6XXQeWGW0ovpg6Z87fnEJeSjXXRyp2xFXWNjC9uy+vh9C8sQnxi
2SruX4ZhGIZhGIZhGIZhGIZhGKatqJ3RB3C+lJUGNZ6Khcq/kFpztF9FcUEM
HI1zcytiJ86t7KpoP/3G13FBiE44b3QifgkVkYd2/LeqzuQbMRQXVP+0UpxP
ygEnEG9W3kXneFpljPanl3qeY6qzU3FOplKO4f8MnGtmV6DeoVonk28at3v+
n5E42qwwzx+74P8Qra5jcN5g+kjFXPINHRc0f/kqnA+oXz2OeHXqbAnqOElq
WuTbFRQXNH/xmc3wH5YbkJ86F+cm5hRSmUW+XKvjAudvjMK4eSgc5xPOj9/G
/1upH5Ea9eTbxykuENM4h3nEouvpe2SzhuLcx/4DqcjXvo4Loklg3K4vnA/d
OCILdZSQWrvItxrEcs/6t+5fCX/rB0vxPFdWZOBxGqkq0v5GHRf4+tl5uK7s
lYnnax9Z9jy0ltSKIl8+nOc5v5l+AvPYvspU+j2PwPdRZC2psXoHfLOI4joa
zuc34pzbafo51Fw+kc69tdrad+OCSJ4KX9kWjUelUV73ND/fSZrqmW/GTqHx
bTNpvM8dfiq178YFovKpTiub4pVvsp8K7RsFEz3zncsTaZ7CZNIDSZSn1da+
1SXRM9+q/gXV1+dOUiuV4l3VvhsXVP/SUfT8u+rXOSnZT41I8qWOC2K8rj8p
gfLeiqe6tSrti/He9Ttd4ij+7tFU59Vj6HlotbRv67ggMshXFT1oPLa/nzra
NzO888XKK2iew4Np/MXhdL31pFL7zoorvOuf1ELfG5t5hr4n9nJXer5lpKb2
jcktnt/fdMQm+PL2l+h7r08v8FNb+25cILKgkr63WldMWryQ4teRylry3bgO
R26PL/B6r+0LldPi6LFWR/tuXCD2I73hWwMGkMbF+amjfaXjArGK9fhzAirm
XEX5Wm3tu3FB/Caa6h5H4076D+ixVqV9U8cFzb++H/mdB9I8T9G8QqsTTr5w
4wKff5i+/l+pTnPu1TSvVkv7blxQ/r19aJ5OVJ9Trut3VftSxwXl5/Ukf3Uv
mn/992k+rUL7jhsXyNlIisvoTnW/G0PzaTW078YF4qzxUXz+d2me8bF+qrQv
1vq88/98GdW94FwzNFP/HrRK7btxgYhJp5spPoLmzR3sp0L7po4LRM7eTfk7
X4eq733hp65v6bhA7G1b4Iu5ZTR+oMFPXd+NYxiGuST+A+FVJOM=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 38->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
H8MwTK34Az2wEXU=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 39->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
+YRUzW9z+kJ1466PYRimUfwIiYocLQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 40->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztm39QVccVx1+iUyWpSoxtxkSatYQYRcEfRWJt2ptRaVVMGhqlRAevGNKm
DbWmxl8x7U0ZK0aCP0cBjV6dSKUKihp10OKKKMbnj0gyIBqZtQ1KihQhmnGm
k0x937PXkfcub574j5OczwzzHc6es7v3cs95+3YvfdNmJKV38ng84uZP+M2f
rh6GYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRiGYZi2PFB24GchuIl9oz8Kxa89zKyM+o7Eq78bNb44Me5Izd2MzzDfRuSO
v5315Y0s/6tr/om6Bw+FklfGb9I+c43v9tt/B4sXI6sr0V659POO5K9RNHKv
L84Yuvvy3eS//PTzI1w/mG8aQrxQcvCmGismKOT59D9V+VQ9VHUUz3vrC8hb
MTPuv27Pv1WwBHHGlanL3drVxunF6O9+7xXXeDsr6Oe69cQxrBuM7PmNHcq/
A95SX5yZP6S1I/HWsrB83/0xr089x/nPfNOwn0n6BHn/xyMNPrWPvorPWWNX
2iWfWgkVTdBBS6+6Pf/2sBjKy6pF3oO32cXYVqy31dMN6Neak9LsGt9rPuqN
mLv2o4Mu7apheh3mU9jcoc9vM3VeLvL//PtB1xntIXd2z8T462M6FH9rHnVv
rOH6wdxryLQ1HyPvfzejBc957RzkuewWjryXfbKQt6rbatfPf8+FTsgLI/kU
+T8qaJ3es+4/+P3wo1Q/3ml1zX9zU/F+5L8YT/k1eUEt+qspwLpAjqpHfRF7
vnCNbw+rcEMJ5r11C9YnnnPX7ij+1vxeSXzLV5dUwqgOff+Q8z7b4ou3Lz7Q
ofplTPMe9sVbVxNP3039UAUFK7n+MP5YciK+11qvjkCe2fWzkOdy2VLkrVo1
mdYDKRc/dnt+5NxfIb/UtWb4G2Io6ocxKAZqjaim3xNLXNf/UkSUIf7FKPI/
0A91w/rKS/Po8jyt20+nNLmuH9bXL0d+7o5HfoguURfwef9Y8b/Qz7RCWrfk
jXFdv9wiNbwK6/zssRUY9z1Sw/tTuv45S0KqH/bKcduxjtl2vxfzqbiMOqam
RbhefwCvq0pfvOwVfxx/j6IpqBtmdP+GO8lfmVVViroTNRH9yEK1jvOf8UcM
vJCP5yu7Gd8D1MkRtG6vf5f2/YZtPBv0+/mXE06hfcikRorrhzwzuxnIF2Pz
GOSP/MFm1/13q3dROfLk6XLku/HkCqhVuhuqSuZjXaKun68LNg9jyOGLGKfX
UxjffmQmVA3/EcVPGRx0/8A8dwn1TZQuoTp0JpH62ZfUSnVsXkvQ+zBr5Vpc
d/8vUb/s3K3UT04c3Y+aZSHVD5XZ7wT8O4+i+Q98jdZjbxwNbf3w9uVFvrwX
lf9D/THSEnDfjexjJZz/THsYfUs23P58qOjOm0N5XuTrH9p4XjOGYX1ppSfj
OZOrk3bC3lT1Xkj9vJy6DfEnn6M6kTgO+23y/KaqUOKt4ydQRzytEagT1n0L
sW9pLkiGys55Z4L1ozbMWIY8eT+H6l94D/rcLrxO+xcjUmpDmsfXnbFfKRsj
af3xiyh8r7EH9Ag6voMMS1iEeVw/BX+z52zUJTX4wYI7yV9z9B+y0M+Urrs4
7xmGYRiGYRiGYRiGYb69iNe2Y9/fGt4CNU+XfELnbaQyjuyG9guIr6nFPpU1
MB1qLy3H/pelVWi752yt+/6XNwLnU57VmVDx9k9wHmlbpGoV2eVx7eeHfLMa
fqJfJGmUt5zOEUjt/mS33qp2fX9XvLv8MPo/FYf3mOWs2TiPNLV6TpJd5JCf
P4Y3TmKeL23E/0HYWZPpfeMlpNavtV37BRDZB+8/eK50h7/5vdYdiOtJqrRd
On7+1z/7mT2Y3/yrtO9qlmGfUKSTeuaS3dR+/li5XehcoLFTIcYbKvKgsaSe
ZrIbq7u4nh+of2Zjn1js/M4qjJ/+YQ6dV5CqHdqu/fwRmWF4b1Qd+Dn2ieXI
Yszb/jGp2kZ2z1/CXN8v9cRNwr6xrPpgO9rzmvC+p7GGVFaQXUaTX8D1e5/H
PrU8uwP9yz1HN9H1aD1EdnGC/AKozSjC36sh9gO0lyXjPstSUusy2a1q8rvX
sLYk4nxLxUyGynXjoLZWx+74BRCbTO2z0qDmn02o0Cq1XcUku8YbGS/Bbux7
hfxOT6M4R/eS3fHzR/xjLLXfeI7GD59A89Hq0XZD+/kjF9P5nno5geYd9ixd
f9dn29g97yS5xps3XqTxclNpnDdTKF6rdOzaLyDeS/fZiB9N8+9Naml17FL7
Bcx/E83PTPglxT1M70sYWi3Hrv38UQsNskfr+7ZlPPWnVQ3QdsfPf/7HyC7O
xNK4M79uuV0du3XMPV72jNb3+zj87fimNqq03dJ+/tjjt9K57H59TrzGaqOO
/ZafH8ZX6WSfatO8MyqhplaVSnbT8fOP39VI58Q5P9TXHUHXrdWxC+13r2H+
PvIa5l3xFFTmDYfaWk1td/wCWEx2Uavja4ZClVbHbixuJ/5GDPW/cDD594in
+O6khrY7fgFkDqRxD1G7fV9cG/WU6zjt54/KH0Dxl0jNdXreWk1td/z8sSY9
Qf6Toqh9xSCat1Y5keyOX8D4C8huF2kNj6V4rVLb5QL3eDnmcWp/pC/FF+v7
oVVpu0h43D2+JoLmN0r7DaBxVX89D213/ALI7E3xLX3ofn33+21Uaruh/QKu
v/tj5D+j+Qv47z0D9ewjtbTd8Qvg4Yvkl7Sf4tdvhJpaDW13/PwxSnNhF1Nz
KO7TrdSfVkPbhfYLiG/IJ//la6k9soz8tTp2of0YhmHuiv8DFmRQKg==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 41->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztm3twVdUVxo/TKo7lIQ+1saXZQBCpNBAGTNEAR+wgyFNImNiQ5KjEgrTA
YBVQas5YaClURqqQgJAcQtMYFKyNKIU8dh6kXCoIIUgIIWxEBBowgCmmVKXe
b+3jeO89E8PNPxTXbybzzV17rb332dy1X+fS49HZk9K+YxiG+PLv5i//bjQY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmHCI3JeyfA2hMvsdkfDit+/qcofZzU+7WtL+wzzbcTa2ecA8qayRnnlj3j2
I9mqvHrj3x94+cmqzz3rdbGNulJ/ubxhy0fh5K+8reQdf5z9w5tOtiX/rby6
LTx/MNcq5p7aff7vt3nHpG3Il8LM9dCxxcewft541xmv7786dKEWfmMulnmW
D6sqRvz0Wz72LH+x+sMW8//zN1GveX/C2XDyz3l77tv+OCdz3vlw4sVnJ59E
/yecep/zn7nWEJVGOfK3wztYn+2JeVhnzdd/ehzf++bJp5A/3co889f55NXj
AfZz358L/5janbD3UajPjl182nP9P1tf0VJeyd6d6/3l4qVenvHfhL24abE/
Tt2/L6z9g4s6sX8v5z9zrSHiNh5Cfr7cvxH5Hvsc1nlnV34D8m7WZXy2/rPU
e/1sOIP8txu3Y58gmlL24/Oiw/gsc5/H/CGGfNdz/jCH936txF//9U/vxrxT
OTQP+frwLJw7rMfeQz1G/eSGsNb/imbML+LoC2HNH1+x5Kk2nR/s6xuqef5g
rjoWXdyD/PhRLe2vTzxyDvl/fjLNBx9eQt46T0Z7nt+NGyJP0f5+NOYH68Fo
xDkzp2LeUOPb4bNKX+a5f5dxqVvh32k06pGj9lG+J3VsoHnAh3pE2sZzLeWP
KruutOTrnyNt7PuVr5TmsZhdjWHdH6yaW4N+9VkZ1vnDyJr4FuI2t/9XWPE/
6L/F/1z2ufg2zT9qwp1FJd/sxnzLMAf0KMT3KvEh3MPJTw/jPO4MzcV+2Ypc
jfVdXdpY7rl+l3V5F/73RiC/xPxxyBPzoXL6/PchyFvx4AjP769VcJju99K2
YZ6RyylPnNPpiJe7KN7uXuiZf+ql1Bz4Z63A+ipKKU/tblmYN+zJlzAv2RGv
tZj/6mTeVuRZ6THazxyrQT9E+0qoKYpbNX/Iyx3wvsHsewLP63zQk8Zj5o5T
rcrf5Jqd/n7IvMGIF0WjKL6hrlXzj5yYXeGPVxFpm+Ff/Rvcz1h1//X892MY
P86MZchj63fpuOcWCx/YhbyvT/hzi3nT1cT9v7XxM5wjjIgonNeNnkWYT0T5
bOSl3alpg2f+jml04NfvIvb7ck4/rNcyIo/mg1/1Rx7Yd/tavCdw7vPROr3i
l5gvrMtHoGZxX8r/g7uPtxQvlrfHPaOsontOtXoKnXeqn0c98vGyVp0/xKQc
7JNUVj3tV0ryaV80MqnF9l3U6VU7MB5b76N9S+Ye9Ec+88QVvf+UKvEfaHdn
Nd3rNqfM4fxn/l+Qnx6l+SLqtlev5HurDvwW+wGRMwvnClmzgNa93D/mt6Ye
e/N63D+IlYn03qG5M+Ydxx5U2Kr197rsLORd9FtvQid+gTj7sRHZV/Ic9ohR
a9D/2K4ZnLcMwzAMwzAMwzBMm1n4Iu7nnB67oTLXhlpalSC7qf1COPsX3C+J
hAFQOf/3+J2Mo1XFk934mPyCMTtu8tE59wuoeDQH523rEVLnbrKrDps8f78v
Vs7EfZlVTP7ylrtwbnduJRWlZDcyyC+k/XFRuB+Q927DPaYo+FkR6vsbqRpC
dmt8lPf7j3mr8P8a1ObM7Yhf0pveOy7V+jrZrfmrPP//g+w1Er+3NKZ+D/5W
xYI3oEWkZjLZjSjtF4Tz7Czc19rZzX/Fc09biPtaMYPUWkd2W/sFY1fGoVyu
+QnuPaztSS/Db6vWbLLLcvILef5EH93TxBYvh98/dyxC+3Wkpmt3/YIQJ/ou
gb0she45lg5Gv80/kDoFZBdHtF9wfG4G7nXUjIxN8Juaj3Fyfp6vx5XsztoM
z/sfZ3AixtkY/et41PMLsRbtp5AaNXPj6fm0XxByfDTql+nTCzB+O7pjnFUp
qfMc2Z2x5He1IV8YcwH9m5IAtUrHB6hrd/2CEXMs2FWXmaTb0ihOq9GV7Ib2
C8YpSCL/3Y9T+dppVM8rpKa229ovJF4lUvkk6qc5Pz5AhbZL7RfCoClUvnoq
9eNgCsVpde1C+wWj2pHdTp9A7fakcbK0Km2X7bzjzSqyOxv0OHZMJb9OqQF2
p8o7Xq4fR3EjH6Z+3JRM/dAqHtB27RfC0BiyJw0jvy29KF6ra3fiYrzHf+BA
Kj8dSc/93u0Bamu71H7BiEOFeD/iZF6GiqjqAFXarmoKPX9/ZjdtoPc7i+6k
dp1bA9S1u34hJD9B9oSDVM+Gbhe+riqe7ML1C8J89xl6v7P3AD3HslqqR6tr
F9rvasNK7tOE/kX0I11zD1StJjW13fULxuxO5XbCQCp/JY7q0erabe0XjMob
BLuMjaV21g0LUFvbXb+Q/l+MoX4a1I459p4AdbTd9QtGVvSndnwDqN+dKc7Q
KrTd9QvG/lM0tXOe/Ozpevy0mq5d+wUjbu5L9T/1YyofTs9pDSNV2u76hTz/
oTuofl0uigcFqOXWr/1CWNGTylf01vXocdRqabvQfqED0Ivsa6ncOdmZ6tNq
abtw/YKQ+26n8i5nPkHcER9UabW0XWm/YJy9VSg3z+RAjYPzqJ73SQ1td/1C
xm/vEtjtmtlUT8ECaleraxfaLyQ+dRr5rRtH5c0W+Wt1tN3UfgzDMG3ifzCG
Uzo=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 42->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmgtwFdUZx5fckgTRqdXyEEhYoAGhtaEmndhAhzNJ6qMgYTJN5WHNGls1
AUvS0Ao00SMkjRbl0ZGEDjJdUEBehvAYC9rm8DAmVSI3JQFqdRYwYlQIATUC
HVvz/851vLvbSwg6Osz3m8n85377//ack7vft3vPvUNyZ2b9KmAYhvnp39Wf
/sUaDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMw
DMMwDMMwXwnOyLNN47qT+Enx/m7lMQxjKHN1I+onM+OgXx2Zb2W/0JX6suuP
HfH1ZR//d6R8ufX8ls7jZsyqt7tTx+LgMMzPmty75VL6gBzV8SfuI8zlinX9
qJf8rm8ZteYQ4psPve97/dc1475s5i/5q+/xh+cu7Yyr1u+843fcLmt8K2L9
1/zhxc7jYsDEU92pP6d/03bM7/UB7d3Jt18Z9yT6xzubmrn+mcsVMf8J3J/N
B9YdhbYOx/UuJma0on7v7edbf3JZ4VG/uJNbV4m6uW0b+oP98Oz3fJ8fCobW
R6or81wyPXf0LPfNvxCqvBz1a+z6c8Q+cyFsMfRVrn/mckMuWoC6d/rswf1Z
Zm9FvZsr1uC1HR1zEtf9tUN9618l/uQY+kPL5oaaz5/3k0AQ9b+9+l30kYZv
++e3p+yu8YmHsBqH4fOHuL/I9/mhywyJP96dfBn33l/wf4if8ualjC8G7NzL
/YP52lETj/uz7FN8Gtd5Syrq1C6La0P91rah/s2Hvvuu7/0/mIznaqvHWKi8
rpg+J8zacgLxaUHct5V9+qRfvigctamz/s3UX+NzhjUh8DLqPXo9+oo9q5Se
P/42tlvP/2blqjcwr4FDfce/EHbNYXx+kMWFvuu/ENa+M/s71ydqXjl2KfXv
rP2Y9zGZLxwn2G8T6q6oAPdHUdCEOhHFqah/J/8buO5lecI/fe/f8fvRP8Sk
JPQPY96LVPcP2pTf+z6q2xF3tfnWf+A+7BvYuRXoE87TJbjPWzdmoo/YVTOp
H20oOBHp+hdNcViHs7sU+4ly4sZ/4XXWPuTZzogu9Q+Rczvu9zIv+x/w503A
vqOa3d56MfUnc6Zsw/ixj6DurV/Ud6l/qNaeBzv7hS2vbqS+Oh3PZ+I/py/q
+cXK7VeNvhPojffHKniuhvsH8/9Qmb89gDo5e7QB122vStxvVMbkiPcdNTwR
z7XW8l6oV2tZEurNLJSod/mbBNSNiFnqu38mh6/dhTzzUfKNSUP/kTlbSSsW
o27VxoATaR7Wqefp+4XvVZP/nklQ68A+6h+puRH3/9TjfbHPqKauoP43/QHM
3y5IofPdvdK3f3nOM7gA9W6OHw+/aCpCvmjf5r9/6s7fsBf7sOb0Cnr+GvMt
Os+05i71D1nVXNJZ9yo/E/1L9WjE+6pSZ9Vy/TNfNnaOsRp1k1+1EdfdtfUz
unTdVyRswHV+cj49D7y2B58HVPxe9A1z+amiiOdpaFqPftE2FfuFTkEJ6lAM
DtK+X86WQ126/2cH6fNQXgfy1MIcqrtthRG/pwxhBYdhn9BMOUL37TsOY3/U
mrOyS+PbMWWPYv3BaOyfGPE7Ub/W61XLLqZ+RceMxzCPJWnVXPcMwzAMwzAM
wzCME037flblU1DzZCbtF50gtSt0PIZ8btTPiul7/j1RUBWVhf1CS6vQcTub
fJ78jzLw+x8RXwJV+VdiP9HUqgZR3NY+N/K1KPrd4pz+8MsHy3bRfr3W31Pc
Dkb5/r7ROVCzG/NtSFHw/6ADvxc2Q/oqxY1m8rmxO2r/jnFb7Z3Q9Lex72/c
TGodp7h5lnxurLzZOzBudU98zyhaeuF7DONNrVspLrXP8/87bWMcOebDKqxz
ddtKnGe91hSKm2fI55n/lXXPYfwr0rBvo8qfX4S8R0jVVRQX0eRzI3NGr0I8
UFUO/+jC2Xg9SWtPiluW9rnzb//vPBz/5twKzP/7a56BL5FU9aC4cwv53IiE
l57F+p/5GPMTA5fTPu51pEYlxY1B5POMv7bPZpx/7rz5OE/sfKxXBkjtn1Lc
eJZ8nvy6Uso/fw7vk5W8gPavRpOqcxR3akt9879yzmXjeztx1S+hzq3hGoqb
2ufGHjmFfBtySUfTayukOi61z4288S7K+3AajdtyB1RpNXQ85HNjLb6H5rdi
JuUNIbW1qqf0a+3z5CfrcVdOpXGWTqbzabV13NY+T/7+n1P+H/U6jljk12ou
0HHt87Bcj7fwbspbSH5Hq6Hjlva5MR9KJ3/anXR8Swqtu5pUpFPcKUn3f/8e
u4nGLbuBfOsTaTytobjQPs/4cR/gexWxJJbW8XKAzlNLaui40D5Pfmkr5Vsj
KK/ohfbPq9RxQ/vcWOO2k39RIumdmWFqLKZ4yOdZf8sMGv98Fun26WFq6rij
fR6ybiXfsPtpvm/k0XhaQ/GQ7+uGjB/5AdablAiVfcdCTa0yFNc+N2IHHbcG
3gS1x6eHqdBxS/vcmIt+jLiTnEb6/gSo0mrruKF9bpx9P6L5pdNxu+6WMFU6
rrTPw700PyN2DI1/5uYwVTEU/8znwu7xQ1pff33+tRm0bq2Gjod8bqxgEsV/
R+cXT1CeejwjLP6Zz4V68gby59NxVT0uXHXc0T7P/K8fRfF++n2eQ+u1tMq+
FHdCPnf+RwnkTxpBvivotdmbVOm40j435uFBtO5rBtN54swwlToutM+NrD97
Br4ZWhtJTa0iFNc+z/yNJsRVkdaoU1BHaygutc/DlHWIW0dWkO+2HVBbayju
TF7nn88wDHMx/A/r6jb0
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 43->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg9wFdUVxpdJQazYFgyGVMAtDIVqASHUhGhwpQg4YBKZEJNYkgWSChia
BrVQLOVqpRW1BAg1A9MpS0hREVFS8gdI4VoINg7/ogmGlOBCwAQQAhGKGErq
+85dZ7JvfQ0vnWqZ85vJfPPOfufeu5t37t69+743LWtSRoimafrnf9/5/K+r
xjAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAM85VgV+fuvS+IPCNy+7vB5DEMo2n6DztX+upHXEn5e0fqyEi6/R+e+bsX
1ARq15wzoMR33Lhn09Fg+rdGLHvblyc3h3v3zzCMpnUdWuZZH7+88xDqP2HI
aa/jIk6rRn3lh74dqL704sIGz+N/SD8RsC43D7DQf6/uzcHUr5g7focvzzrz
7Jlg8s1vHNmE/Lr9VTx/MNct8Q0foI6HXz2I7/mI2m0+tZs6oT5lxLe966d0
ZF2gurAaEzA/iGb9pOf8UZxbEShfTB39Hu7/enhQ9StK0//kyzOXDP+oQ/Ub
F3qA65+57li5AHUvOuXX+1QfUIp6N61s1Is++cLH+N4XvnDW6/tvTHi5zfpd
Jlc91cY3eW8j6rfwdFD1q8VH5mAci25u7Ej9WbVPB15nfAkyc/mLuB7be+3k
+meuN6xvnjiC+jxadw7f8wfOU51XZdPnielU/1PnN3l9/3XtynmsjweFH8M8
cuteG+0NnXYc64a8nsgTDf/6OOA6YXdhm+cHveXDfci/pQvaNceO9ez/P6GH
vIR27YgTns8v7aY5zXP90l6MqNhDHZq/BrU+u6MD+QzjhTCfeAP317Rk1KcR
/yvUvWy9hHqzXnoTKvU9xzzX7+N64P5v5mRfgK90GM0fnTPQjui9Cp/lxn96
rh+sG/ZvQb6WgvqyUi5hPaLNXov+9AcSsW6w9OjzgepHpmyp8KyPHkewbygr
7aDmDxn6xvuYP3YmB7d/UJ0tkZ9yIaj1i/1U7gHfecmW6vqg3n+0NLyF/v+6
cDmvXxg/0mdtQN3vDsE6wA6tQt3Z3frh/q1t7I36ERXN5V7fH7vgEdyn9XWX
ab3Q30Kdm58WUd3GZKDu9PLve66/RVlxIdofXIn7s/xtOeYBs3cr5iO7tDva
tVc96nn/treeysXxFfQcIl4oRL8iaTHyzXfeRf/G648HXH+Yh7cX+erMmHGY
zjehgsaz9lXsW9rvLA6Y7yBX3UfPCVXFOF9R14jxmKcLTrUnX593qcw3DvGb
OtqvuHsHrqch3/ScP/3O40rOrh1op7YY/U6NpOezPa/ye1DmS5H9B+M+ITpn
oB5Nu/wv0ANL/xywbmKehN9qDStCvaRm4z2bFTHlMNrLH4X7px6btNarHfPM
HStwvPYW+MTqKFpnrLwX9abvfQ/3TXn804Dv7/Tpi/fReuP3yBO/7knvC6p+
h3WDvetUu/b/zP1r4bOTuyHPWDQd849x9pX21X9OOPYzZcwcWkeVXkU7Mmxp
u/YfjNtS/obz7tsH86C9YRjmD73/yNprqV+xbudu5I+p34/xn0t+nuuf+X/B
fnEg9u2tT7bmX8v31lqcgzxjRRjmJdk8A/UkJ28LOI85GAdtvG/UWm7Gul3v
Mgn5xmt3b2vX/btgySKMe2I5xi1m99iKz02xS66p/qKHLkBe4kWuW4ZhGIZh
GOa/hn7DBPyuTYyfBzWW9oHKHFJrHMVN5XNjrE/E74P1jBr8PsYOux1qKbXT
Ka6/Tj439syT+F2xOea7FfS8n4P9M3MPqRhNcaF8buRdq3fR87kFv/5YE9bt
xgxSeY7iWgT5/Pp/7CdY34v5c+l3gnllWK/rjv6C4sZM8rmxJt64HfHni/Ae
Q/t5EfZNrDmkchHFRazyubmrpBTHK3+wGXqqZD18x0iN9yluKJ8bUXyE9mk+
67aR9h26roaOJbUvUlwvIZ8bvbQf3v/IRA37M8alec+hvSZSkURx+Rb5/M5/
/cN4XpJRqfM9/z+jKG5sIJ/f+EfelEnXLftp9Bd2fBnG3YtUi6K4NkD5XJip
2hq0n7gwn/ZbynCeokTpgxS3E8jnl985v6BNfOJDzyAvivQLurh8CiOrBs+V
1ulh2EeXz83HPpr2DKk8SXEjs6Zdz5//a4zbHsE+mT47Ayq6ToeaSp24qXxu
5OVUxK1ZaeSbalI7SqWKW8rnxuo+jXwfPk7Hk37aRoWKOz6//mtUf5Hq+Jop
UNsi1VXcVj43IiOdxr0zC2rMy6b+5pIaKm4pn9/4s+i8RPws6q9wJn12VMVF
lvf528sSKJ6n2vnZ/XQ9HH2Z4sbyBO/8W8fQ+GMM0tZoGodSqeKW8vldvz7h
1G9cP/KlDaZ8pU7cVD4/6vtSP3H30DiHjKV8paaK68rnx6QQOr7yLO23XmyE
mo6quHw4xDPfjKbxyZs+It+ay23UVnHH55ff4yj1W1sL1a80UF4LqabiQvnc
2PsKKJ7yCvX7Whn9HkbpF3HH93WjeAje29sDf0Tv71eMhhpK9UEUd3xu5KQR
iIu8aPL/cXwbdeK28rkRB+6neP046n9yAtRSKo9R3HR87vzxP6bjFQ+S5ie2
UaHijs+NlTuK4qnq+AfxNN6DSqdQXDg+d/9P3Ev91dL49Nw4+qxUU3HH50bP
HEnnmWeQb+FD1I5SJ+743JhLIqm/J2Mo/9EJdP2UGk5c+fwwBtPxxOHUTyj9
3zVHVdxQPr/r962BlLfpTjoeodpRKlRcKp9f96N1it/Rl7Rnf8pXaqi46fhc
yBuvfgJf4mdQLa4TnX9sJzV+ijs+N/rsajq+8BBUZDZCLaVO3PH5jX/dNvJt
KaG8tEr67KiK68rHMAzTIf4ND/lDGA==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 44->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1QVlUax2+SFpqt9mW2itdPzNIaNwK2hJOWyGpmaqmF4xnSSayt3G1X
XA3ObrZb2petGhLqjbD8BnE0DZAjEoJg61aYK34cQkESM0yTbQZc3+c5t8l7
r+/gi7M59fxmnP/w3P9zzrnX+5x77nlv9/hnR08JMgzDPPevw7l/VxsEQRAE
QRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRA/
CeKG5z6ODiQxq8+OgPIIgjDUloSdl6J+rBeMz7zaUUf+/rnf9kOHb/Id508N
PBDIOPiMToW+PHZ20L9bch6sbM2LNI8QvzTMhGn7ffe9irj+a6/7XxpY1yL6
9m1ex/m8zum+uHlr/lHP+j/ceNhfXbGQ8L/m+9qv7Xo8kPpTH1RlwfzxUHhA
+bJP+QZfnozY+C+qf+LniuA94TnMyx/+9Mf3OUtcgPW5tPQrz/s/bbbf5zLL
6QXtqfCVx7x81oTdu/zlq9BYmF/4e1fVB1J/LDzqb1C/L/Xz7L/ZdAi+JOsg
grisaNMAzzU1Jr0S7u95BxT8rTLw78wnauD5fmiW5/NTNU7dC/7K0GWe64PY
VjA/yD+P9nz+i6y33m5OXYkMVt2S+jMbVlRR/RLE+ZjJR6AuROEhqG++5Dio
yC+H9b58tzuo9d/Bns9fXh6Kdb3jWA7kN0QUQN7Jst3QTkUStC8PdPvGK998
cvZieD+P6rY53+M4a78W9u/k+mkBrd9t1O9C6gLK7xm8CPJ2R7Vo/pAbR5TS
/ENcbrDcw3BfyupSqE9VXQgqpr4OysNqoe5Vu9O1nvdv3JuH4P3+8VzwW61u
A5+Z+CruF5Tq+WPOes/5wxq6PdtX93xcOaw7rF3PL4N5IHbwHpgX6t+B9Qf7
NuVEIPUj+r60HcYTWxhQvtHxC9i/YEOCAnt/WL20yHc+7OWwFs0fvD6Z3j+I
S8/8pblwf8c0wHpf9kmH+1x1TsPn9S3x8HxnSXl7PNf3+z6BfXWxbSDOF+v2
wXNayo5Q92J6EcTlm6099w/Z2S7roD6LC2HeUM8cxP6m/APq3vzjnThv1Pb2
+/yXKZvzwd++HY5z9DrYzzDjB2B7Byd7rj+cqINvwHpEdJ8I+5n8bDXULVux
1Xv/4wKIxNOWr+6thfNhXuNJ07znzwudz8ih8Hsnv2PUl3D9BiVfVP4PfJ4A
+ydm2zckzR/EhbBe+b4M7o+Rm0pAb+OwLmA5r5X5u2+s8iR4vgrrBMwffE4V
1Am/ci2omlUEdWydXv+pVzty9Rd5cH/2fP0w7jswmCf4JzGoU+vgua267vvS
b/0HLYR9CDMsCeeho0Ow3itLcD3TOeuk333GPXUpUGdx0VBn/MYgyDM/zMZ1
0d0dPOcvF8MSoM7UsVrws+y2MH9ZqfHNmn/kO6vWQ//3dcN108T7cRxnptdc
1PzRug387mnVTYDrLp7+SxHVP/H/QjaVvAf33WNRCy/qvo17dAn4N57C3yMy
skBV9Pcp/toR/fovwH2GZ/B+Tz0K85F6eTPsG4phIf9pzjh4bSfYt1Bjeh2B
9r4bDOsHcYVo1vcH6uZ0/B10VwG8F8nfjMV56+6dey/qOvyqGH4XYXOvxvMP
a0oPqH6bJmVT3RMEQRAEQRA2rPRWWF9afBSoSGvA73S1ykk6rn1ORNcw/K52
ZSasm407auB7AjEA1Y6LkDDP72/NP71fDOvb8QWgrKQ37CfwYlRzHMbF8+hz
wptGw3uusSQG8zq8AO/fvKPWZRiXxphCr3yrMhh/r5y9fCvofLkF+tPKEjFu
VKHPdf3Cp8D+hSyo3QzjfWsVrLfFAq0S4ywSfS4ib4LjlnwIvjMUZ95dAe18
jaoKMM60z4lI7gfHrZhBa+B8wzcsB18kqjkE4+dekDZ47r+kvr8afIXJmXAe
M+etgn5noAqJceOf6HP1f3ZHBviOt4f3NjZxLHxvyR5HlV9hXDWhz8WWLpMg
/tpn8L5oLEyFfo1FqHIOxlmq9jmw2u2H85M5I2CcPDPtI9C1qGoNxs0g9LnO
P70Gxzm5NRy3Ti6Acco6rRzjSvucqOWL8Xw7HcL/h7Q2G8G/CFXegHGWsdgz
/6eG93gY98UqHwE1t47Fv7WaOi5snwOrcSoe3zYDj8+cCcq0WjrOtc8JG/oE
+qt0O72mnaeGjts+J+o4xsWGZ7X+HvvTauq47XOd/97J6O86HVTOfQ7702rH
hfa5+m+Ix/Y/Rr/cNx7zK1DNIozbPlf+zjhsP2YUHk8YiflaTR03S+O8r9/N
vTF+8F7UvlE4jlBUU8dN2+dAnLgL/UL3N/xa/L1Fq2XHtc81/sYueJ2+uQv8
MgK/07S0Mh1nTV28+4/pgecb2R/HUXdN/Y9V6DjTPhfHfo3Xr7g3+Hn0GlBT
qx23fa7+B7TC/pIt3Gf9QzCOV6sdt31OrIo87LepHvNT3kafVqHjQvsuN/iZ
sFMw/vH3gLLTsaDqFKql47bPiXoqAv03RaO/ZDj6tdpx4+kIz3y5/QGIy2D0
W33HYb5WoeNC+5yI6Rg35+p8Nv48ZTpu+1zjf1Dntx2B48ibcJ4aOm77XP0n
3of9XT8M814Zq8eDynTc9rn6j8Trw7Pvx/yGUejXasdtnxNrVjj6Nw3CcXz3
ILajleu41D4n8oGBON6c32I7u2IwT6v1EcaV9jlhR27H/KY7UWdgf1yrpeO2
z5Wv+mBerx54fFV/9GvlOm77XOc/5TqM512L48wNwfPVauViXNg+Z36nmm+h
n4pqUFMF6f5QhY7bPicicz/GyyqwnZWNoELrD3HbRxAE0RL+Bxi2VtE=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 45->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1QFdcVxxewjcm0lGKjjYBZaD6LRg0ZrbbWzUAbpyFpVEQgNCwYTWwU
bTSkLSG5aiQmTSatEkyLyjqKNSqpRqxoiG4rBEgCVYT4QRxXPiLGCPLUmjKN
1nf+dx15b32BpzPWzPnNOP/h7P/cu/e55+69973IjFkTpgYpiqJe+Bdy4V9f
hWEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEY5rpC27/xn2Ov9U0wzPXKDQn/uBr1YwUM+tCpHfOd3H/5bD+7ssR9XVgb
PvbnPtSyhPfdedap4e/zPMAwvcOMuPWgu27UmOPHnOpHHT9/N8VH/tDxPavO
DF7mjpv31Dc71v+44BZfdSmSi5+h/p9afcKv+u0fsY36//Xh4/7kG4WDNrvz
tIzt1Tx/MF9XdONQDdXJhFHd3tMibAjVp26tdKwf1XrgkK+60E6m1bqvG32W
dTr5RErWbl/51pdGHb3/XXee9Kf+9G+FFlP/wbv8yrcxb1q3h+uf+bqhRXZ9
RPX1Qtx+qvO1+aRicTmpFbsf9d/3Tef6j2qsp/pybXBcX5snIiyaV0Z/13H9
oNTl5fmsq86GxfT+DYtovaL6m392H9cvw3RHC1E+pbqoPfcZ1f3EKPrb3Pgl
1bs+bA2p+Z0Ox/enWNtO84MYWdSw84JaHbPf3nnp9ao8zCPpM53X32/GJu10
ittElmx3X1e3bfjMn/rVsgNmUN4tz37q1/lBW81sGv/rr310JfOHlZ+xwuc4
GeYaoKVFNtD+OnUFrc/1pVtJxbnMDoqXLUXdx8/pcHr+rfRhTeTLCKX6MmLT
D1P+zcn0t6gupX270T7Ucf1vo/50Z7d9h1lSQ+d+xtkP6P6swaZj/z3FCrjD
v/MDm35jPr+SfGP0S36dX9qI9F+u5vULc9XpMDZSncWpbVRnjXGo9+l1VK/W
wmZSY+COI47v/9+Oo/25XnGvi+aBRWfoPa2nHKF6NbuCEZ8S6lh/WrPyB/IN
nn+U5osb3ztAfrO5AeuOAe1op9Tn/HGxvcPv1l36njX//jKdT2h/+Y9/+/9f
dc5zt2duW9PuV/7gu7bQ/W/K9+v80ca6vdzx/PSr0Cu/9zeeN5jLod2V8hTV
x4GjWKc/ehvViwg9Tft2fUYa1aPxi7wyx/3998/QubgVuI/qyyjMpbrXxmtU
L9o3y/H3rZubnPL19a+vp7y8BHq/qg/1Q53d9wj+nptF7VpRe33Wn55aQ/cp
CuJonlHrasmvZr+CurO6erT+13OSaN4RpXNwXlEQgn1QYkyP5g9908Iqmn+O
59B49ZyN+BzWzm3rVR0WvUTzobr7Sfr89OLCXq0/rDuW0LyjZUXTuYfYUV3J
8wBzOdQ7l8yi5+Q5ldYDysCCt+j5WbfF8PXcaL8Jfpqez8UttD5Vc158h/KO
JdNzp0UOoXNzfV7+cp/P3337i+j63XEHse5PoHrR9qZQHRjFsY7rDxs9bDn9
TsFQR1Ddql88jvXCrL2o27dSevT+1Qf0oX5E9EHKU2ePxH5o5twe7R/EqQfp
e1D9RDn59a3vYh01rbJX5w96UCHmy5pi3PfkP/n8nuVyaBUptVz3zPWGtuiG
Aqrn1Pri3jy/5qhmytNaz+J7+1dSTKrjtJd7tA42tuRjn12pltK6/Sfbd1B+
0piS3tyHGFCyku6/cj3Nh8YngQv9qsNV2Qu4fhmGYZirjVBC6Pt7PWMEqRLe
tJfWu2FQIx1xU/o8MZ7oT+t78ePF9DseK2ELfs87CWrHxZP9HX8/oyVmVlH/
Sa9W4foxWj9r59ugiYjrkzKrHM8fm8LLqf2pIbvIf3oU3vdnoNYTiFst8Hli
FeyhfYNoG4X3/OYh9HtBRarWgri+fI/j76D1Va10LmKdTC6FT6H3vWpAzXbE
jaJW5/OTmAVbKT+6DPumtWP/SvkroOY9iFvS55X/fJHcb91N5yhm5lBavxiz
oWoL4toL8HnlNy9cR/6mYW/T51Bo0r7PXA4VhxA3G+HzRDt/+yq63vHfP1M7
z3Rton7nQK0TiJvn4PPK/0HuVPd5ie7KpXWbiMymdZMZBbUsxI0Y+Lzuf9AQ
fE6rV2O8D+dinPFQkY+4uBk+r/4//gbtd7WlZbTOFBmT/kjtJEK1JTIufV75
fR5Du9Mn03jNo0OxXv0EKqYhrgQ85tj/tcaqTnbhuUmFxqS58HlDlVmIKx/A
55W/JAPn+5FToYehQqou40L6vPKHz4Av6nfID3+um2oybvs8MZqmwWc8jXb2
ZXVTsxBx2+eJ+u0piKdmQodPhn8YVJXxiz7P+38e47LykzDOwET0K1Vdirjt
88Rs+Bl8yePR75FH4Lc1CXFL+jzRKmMxXtcDuH4Tvm+xboQKGVelz+vzGxuF
+01upXMKc1mQ61IVMq5Ln9f4A/qi/TH/xnnLvUORPxyqyLgW2Ne5//h+yD+w
C76Tt8EnVZdxIX1e4/99NP7/7+/EOUvteYxDqh23fV73/14+xneqBXlTAnE/
j0MvxqXPq/8KeT0tF/1rjfi+zFYZF9L3/4aY96PTNN6oMaRKWDypMRCqyLgh
fZ7oD+K6Ni8O7bw2ntR6FarLuO3zRMsbB/+ih5Afl4y/Y6GGjCvS54lZgbiW
A5+xIIXUlKrLuO3zIghxtfNh6BuPoj2pih0Pcs4X8bL/ufBp4ejXkKrIuO3z
yp9wP8bb8HNcf3Yi7jdrYre47fP6/CpGY5z1uK6vHAG/VM2OS58nRq70jwuG
b0THKcqTqsq4KX1e/Vu3IK/zc/KboSap0k+qC3Hb54n5YjTus/oI8sNrkBcB
FTIupM+LuC/QvrGH1DpXj3akWoUyHgufJ3rXGorrtdNJjUNvIF+qkHHb54nV
WIrrmwqR/2Et8qXacVP6GIZhroj/AZHuYHU=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 46->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1wFdUVxxcSa6GjECh+AMKKBgXkY6AoLSa5IYqKfDUNFQXMEmkMiHx2
JgoCa6JSbYpRQL7Jhq8hAgHUVgeBXFAGYiAGQ5Ag4EIgPEBJlFBRa6z5n7sM
2bd5vjw6ltrzm3nznz17zt67+/bce/bu3pw0Pv5PYZqm6T/8mv7w+6XGMAzD
MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzz
P4V1ZtiWmP92Jxjm/xy7ecomzzx88bX8QPlpzXl4Q81+I3JJQSh5LL77x/uI
n7D4XR4HGKZ+WLdmHqzJG1mYX+6ZP9/oyEsZ/9G2gPm1PvxTr/3mfWOOBZOX
do/0M6Hkr3XdOvRLXCgIKf4iLdvm8fjB/FyRlUN3IE/aj91+6X0urPwy2DsV
euaP3vnQ4UB5oZf32VWz3z416ksvPzE+7aOA8/+Uiv1o/6aMilDyT39hgqyJ
M3+97ItQ4g3f9gy0X7mhiPOf+blhjuyKvNfFnajD5eGvS3Cfj87bie2I15H/
1qTIzzzv/8yrd8M+sPU6r/12acRR2KP6nApp/p712Frk38aUoOoEP/7VYjry
f1vbgOPUj9K7OiXvcuIZ5gpETDpyAvk/5AbU92JvMfJVz6w+Ce34nQ/50yvN
c/40syLLAs7/++/B87eW2tR3Ofkn0zqcDil+/MjnUH/8detxnr8Zpjaygw/P
5Uavq1BfG4UFn2O+vyXxLPL+7h1Udzf8vtKzfu/zlY38OjaIxoHvy7die1Eh
6nqxNBnjiVjfzzPewb46Jtdrv5E2ltYVlj7vXX/8CHrZvDSc1y8SQnp+MBsf
egPncyLt6OWMH2LiqN08/jBXGvLWUfR+7LUeyHvtkT6Y5+XJPchXw5dN48EW
zXP+FVlTipEfn7aCv91jOOoI8+kvka+yeCHyzlq6xbN+kPErx8E/K/rApfvt
3velUz+SMW/LFzcFHD/qZElJJtUzOSGNH2bEqcKaut9OWXEylHh74VM7a+L1
goWXVf9YOzd+zOMH8x/nQseXkedT96EOMNf0Rf7KPzdAvon5zfF8YLY/X+o5
P1dn7IG92YdnaZ0vD2r5KF7vF428t4bHnvWML1n/Fuynx2B90Zj4CcYZ+ehk
5IvxzxY0DvUe7Rl/kd8OQN0hZ1h4XyEeSMZ8a3ZfQusG7yz+vD75I1el432l
HNII44/4uxHU+GPHdiipyXfDVzAT8dXhVP+UZtVr/cM+fhT1k/5US7Rv77BD
ev7R263BuqWYcvtbPH4wdSGKZmO9zy7OxXq9Hj0YKm88H/D9vJzXBfeVnvck
1tf0ve/iecBYexvy166Iovm7UWLA+cs8FYU6wjz4Fe5za+vfKun5I4bqhk4v
Bff+rnca4qwFEZSv07JoHBqXHFT+Wql7j9F6aEM6zsyZaN8oSQ48/ijknJFY
7xDj1sJfT8yg+JzIoNrXl2dnIW77DbgOxtznqa66Zpn3+9e6jlM6dTXibm+H
6y413yrOf+anwm6d8izyp3P50PrcdyIpbgbyPuwM5i2rb9k+3L+bpi8Pav7t
NhZx8sAs5LG5YjrqF+3xqkPBxIu10agbrDataBzIpvVPe/C2oOLtb1JQT9ld
J6HfZqOMI8jjIUsP1Os6PGEsoPxtjTpAhv9lUUjzf6vdGznvGYZhGDfWw19T
3Z3bkubZX+VjW1cq15HdVn5uzPnVmG+NBtOh5vDZH16qF+0Lqj2/n5HZAs8Z
xguJ9J1Qzw/o+yOlxnNklxb5ubHbn3sP+1ecpLiiJvS9j1K5iuxGB/Lzaz8s
B+8X9BFlWAeVZwvfQbxSayjZxVU5nt832gnpm+H/zLdvQ1/Nxnwr55KaU5T9
IfLza79rG+yXsf3pPcPiNqjTxSuk8h6yO35+vNwY+7X9u3IQ91D0cuo3qdhL
djlL+bnpexvt/2QAvt8QkxbhucGcoHQ/2c0o8nNjtj2fjbgZ8+fhOq/LoPN4
ndSaRnah/OrCumMrxR9pvQTt2aSyFdnrQh/y2ErsjzmK45up0+g90mRS0Y3s
en/l50J0GoDvS4yJ6bje1riNqFvlH0m1J8ku7iA/N7LlZ8tw/JnrsV82rSL/
cFLjWbKL68nvSsNemYjv8vRZj0OthBSo4aiya6sSPb/f0+YmU9y80VDz2jHk
10TpfGWfk+wZb3YhuyweT37RE2qrshtdvOM1g9qx0lPJPy65lurK7vi50WPo
u0RZNAJqD3qUjqfUVnbHz4/dD1D/jv+BjrN6EPnlkJrK7vj5MTmW4jeQv33g
N1BRSqoru1B+ftcvqSP1NzOO2l/Tk7aV2sru+LmRbRvT8ZMGUnx+PMUp1ZTd
VH5uxOZirG+IzddQu+260/ko1ZXdVH5+vHmY7M9UQPXyJtSPE6S6sttvHPaM
t166QOsr4WHU7uzf0fkoNZXdVH5u9D27aH36g9XU/rDt9P5LqansmvLzi/eV
0fmnHqd+hlF/LaW6shvK70rDHtmrCv32RUGNLf2gmlKp7FoS+bkxmseQPffe
KroPfl9F/zupcOyOnwt9UH/YraHx1N6bI2hbqansmvJzY614kOxxql0xvLYq
+0U/d/8Xk13GDSbdM4yuh1LH7vj5Xb/J99PxIweSv/YItduAVCi74+fHg3R9
7BF0fDs2gY6n1LE7fn79P0H/j/xCUD9v6UbXTaml7Kbyc2Pm9qR++hpTv7tX
noOfo8oulZ/f+bdoRvE3noG/eP89qKlUKrt2XTPv/+/tznSe91aQ39SGdDyl
el+yG8rPL96IoPOuPn2O/vfryU+pruyG8nOj5xVRv3NeJb+ETbXUsTt+fpQc
JP+7Csh/57e07aiyy30HveMZhmHqw78BlCZToQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 47->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
ie1fz3Dcnz6Kv4dRx66/H+k+f4IgiLbwH7fPXhg=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 48->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmgtwFtUVx1dCFHkYbDJpLI+uY51Bi1TlQ4GIWSCImpCY8DBRGZePopDY
pEkr0yDQi85UYo28rDwMZJGEh5EiCQaoEK8BBiqVUBQpqZQNIaFAJAUSSpVI
zf/c1eT71hC+wNQ65zeTObPn/s/eu1/23Hv27t7sTU+cFKRpmv7VX/ev/jpp
DMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMw
DMMwDMMwVwRZUbs56iqeX5x+ddfVPD/DfJ8Rrw0rb8ofK3rCp255JI/MlGjP
qZSueXZ86swmvyz4/W63dj040m4tP+XqDU81tdsX764JJI+tt+tKmuKMReeq
eB5gGHeM0ttK3fLDiJtXjfxO6PEv1/aCSZWt5ZV+uBjnlTOyT7rqXp/wt9bi
RbfEA03twltaF0j+2uu770T8y0GnA5o/Lm5Zh/E31nzE8wfzfUWsTduGdbLg
tj83v8/1o3FHcdxt4ym3+9/cvdx1Xf/6vK8sxPwgO0xxnT8uhYyoXNIUp1/8
U21A+V+yaiPijIZj7clfMa7Wvb5hmP9jrJ47URfbQ0ejDrfDnsU6Zw1YfAjz
Qe5y1N12cb7r+q2fGPgPtPeSr7m2DzqL84rpQ6vbkz/Ghzmt1hmXQtbH7eX8
ZZiW2BOTkNfyUMkJ5PG7vbFOy4vncSxmPYN10/jQcq2/raKQT6Dr01iIuAVD
dr7nohMNN/6zXevvZz/+LJB4Y5q2DPNZBF0fwzDfIDNv2I/86LyP6vuHn0b+
ize8NC/8+t/Ie+MH97rW73ptLOoF8+R0ek6YNXloU/6L1B8ehD/jLdp3y4pv
9fnbSB2W6tYu5hrYv9PG1AVW/8/Im4fnh5DnA6r/jZHpBbj+mIRD7apf+nZs
VzzDXE30c8FYH80LN1CeBudTvr+chHVXj69y378LEqirLZGCOHvEfNIv6Yb1
XvxmHY6tjFdd5w/7ns0rkV9hGR+jvVPs+Ob1gxneD3W/1c8IaP9OH99rX9P5
rMj/BJT/5uz4PRjPrsnHA9q/yFq/A/Ph5mUBxTtYMo3nD+aKYz677U3k7YJO
uL/MOTsp3yoHIn8Nzzx6DhgSVeGavx4v9gvtJzoivw07C3WE8dAdqBvk/V/g
2M6ucl2/jcfXoT43D5QeQdy9M+g5JLsI+SrjaR4y8ue2af9Q9v0E45S3Bu+D
7V9L+waF2mXNH2Jj+eKmvNV7j0Bdo+d2b9P7ByPRi/lGK/b8Bde1upzen1Sc
cp8/vwV72gHMq+ZNvRBnvOO5rPivyZ1SzvMGcynMnPw/Iu9ejF6D+7VuHPJa
v+7Jklb39x+Jxf662aG+GPNE1y54nyeOLcV9b77pQf5YY+tbfX9mVgz+ALrl
qbRP2H8U5X3drchb6+yCNq3fsnwEdNbxUMp379YzuK6xG9r2/iFrNK5bhLxE
z0EZExBvpuuu7z98sTc8X4ZxZ+cg3k5JpnlwwqC2vb+MyJ+LuG6RmMesx0PQ
r1h4Z0D7p2K9dYDzn/lfIeKGrw7ovp3yINYt+2ASngvsqLVb2rT+5ux9H7oh
n+N7JT2vCPsP8vY7/96m+PrYTci7hsOoh2RyJM1fi6e5fv/ki33TLb9D3Npf
/RX6iUXoV9YVbLus3+G+Ub9F3g+Ow/eR9vYvczmPmSuN3rCD1uVrj8OK0FV0
rKwRTH7T0flgd6xAnSpWPEn7AJ4M5K3sFYpjo98meu92bYXr+zc77Xp8n2vN
iabvdLsEIU9k4TtYR83CV+CX9lT373g7byL9wWsov440vod+Iwbjux+zKhHf
J5hGyHbX54+SFMwXIuL8VrQXxSD/tR99Ab8Ze2oHzr9o9vuu/Vd3wLwk97yB
OkisG78eurJbYI0eJ+j7gwtrXOcvq3gW6itRuh16uTAT+40yoQLWXDGkCMdK
54tI7kv9ze6JedZeOdzC8Rqy1gvkF0lK54OcmrcW8+SuPqugt9IRZ+SRlWXk
1zJJ5/f76Wfegj7sXYpb1hv1o1hE1gwnv/YT0vlxdxbqTrnvNHT63qGof8RH
ZI3d5Jf9SefX/9YEiv/lQ/mIs2qgN3PJClP5ixPc4491xnsrc753juv9+QL5
ZTXpfBGdk+bDH3l4BWyfbHoPHkJWeshvdlC67xqjVH1bMwVWn55C9fJzKS38
VizpfLH1VPJPTocVcV6KV9ZUfsPR+caHkd+Kf4riqieRroasqfwyzD1ee2YM
jbv7OOp/6Vg6n7IyhPyOzhfz/BOkS36a4lZF03mU1R4jv6PzRU+LIn14Ml3n
1pEtrKn8js4vfkYStR/9OY2jB12v3pOspvym0vliJBvU34t9qL1xNMUrK5Tf
0fkixpzDc5L5+qf0nLWy75nmVlN+U+l8sQZ9QO1WV+o3uoF0jlV+qXS+yJmF
tG9c+jGsMbEf6b1kHb+ldH7jP1ZGcV09NN6fZbawjt/R+fUftAZ+/WQVXccv
wik+jax1gvyOzq//njbFR16AFZlf0nmU1ZTfUrrvGnbosHr8TqEPkC1PgJV7
Elr4HZ0v1pbh8FueB8neldDCagPIL5XOF2N8HPUz6BHq54FHSa+srvya0vkx
chTF5cbDipWPtrD6UvI7Or/rT4ghXQz1o48dR+NR1n5Y9Z8Y4xqv36+ue4k6
z9uJsKZjHb/S+bGffhdZNpLG3ZHiDGUt5df3u/9+ojKKdPOo3RhIelNZQ/kd
nS/mNTfT+XfdRePIH9bCaspvKZ0v8vaqs+jnp5/Dysx76ptbofy20vmNv3E7
tXuOwoqZOvWnrDaA/LrS+fWfR+2mvI7+X+H3Uf/Kasrv6Pz69x6icf+hgfpZ
ElHf3NrKbyqdX//PdaLfuwv1Y954B123so7fUjqGYZh28V8APD2y
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 49->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1QFdcVxzfTWA3WVkJKkYRkrZbGJLbUNiRW1JuEhIBWELHpxNGsHdQ0
RutHrZQ6ZhPHpGrGmBZo8AMXRCMGRZJIlCTDFaxg/YghCgFE108k+AENxrEG
0/A/93Xy3ts84dGZJPT8Zpj/cPacvXf37bn33Lvb/7e/T5zyLU3T9M//+n7+
10tjGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIb5RmBtvKF05FfdCYb5hmJV9Slvzx+rPuB9X3lk3/uZ9HVcXx+U
7nRchEcd70h+2vf87Zg/eSzjX3ulPU5WZHWonS9DTH5xEY8jTHfFuGfpdsfn
u2XQKdiT7IuOx98ec9JXXlg3tbzeftwcv/e8k581cEe9z3Fly8rDyN+sG5v9
yT+9f9VBjF/Zp/2K1w6+vxXxNTP3cf4z3R2ZOcmtXrbONiP/ZVjIBafnX+p7
KnzO+3YGxgcrPNt5/LgOVv2yIowf2YF+5a9YumJje5xefMc5v/LXfAz1g1Ef
VMb5z3Q3rIqEE3iug8s+RJ7Oegl1vpwgMG8KGXUGGrDFMX+s2x6s8Tn/xzyP
/DdnfnqmK/ljnQs77df4MT4S+WsGlv+jS/V/aGkG5z/T3RAjt2NeN84kNVGd
/SlUtK1qgD28jfLu5RjH+d96IKwacWn9c794XNT3fM7N/8MlZ/3KnwUr/oT+
pJ32uc64HnLrNr/GD4bp1kRexryvT78L+S0efQzrdD1iIsYB+9b5qNvtdxod
62+j6Cf/RN0QOaEOx3Paykva7d9fgP0E/RePYP9QxM/3Wb+L7wW+UeJgN4ek
r2i3y4g8v+p3sXhoNs4rtvm1f2jU5mP9L2JoHeIvprXK5zqJYb4K9CWxqcjT
79z+EebxU2XIU1lnU75O3w8VQZcd53+j9mwljg/Ogp+VkoJ5Vt57CvsG1ozj
qCP0482O+a/37vUs8vvcE2+hTp8SdAD5euPsFzDuXN2A8+tvpTi231FMI6tL
6w879eKJLsWHb+9S/WIGLtrP4wfzv0ZUPYl5TSwbRc/3pQLkmfmHk5j/Lb0v
7dtdntPg+Pyt2Yh9cet4LeoG+a4Gf12n/QIj5h3kvfH3K477f+JHAdnI8/JU
G/243ET7BeU/xT6EfWV0I+KjCzu1/2c3lB5EHTKsZj3689p4v/YfZew6jEPa
6EOO7y+uh/7Zc9hPlVvWNHUmXhzUt6L/hcewD2MWpfq1fhGrz3epbmH+PzAb
+9B+34g7DyDfombsRF4v/XYVnp+2PW845u+7IdhX08e+Wou47/4M63y54YdU
NxzIo7ztveSoz/cEoaHYR5Q1oRh/7OB+iDMvPf8vxN0S3KH8FT3iaZ5/cye1
2zqqBf2zd7R0JN6snlPannci+gXkq5lziMbDOLNj8ckP5bXH21cSMG7JxQkY
N4yBsR0aP/SqgW+i/aQsvBcVcWk0fubc1an1j2WIHLR/bT/WZVbO/AU8DjAd
xVr/4kLkTd+G5E49d78sngv/FR/PxvMnnqH64O6cTu2fy7Ji1ANaXjHVzVFX
D3UofuozNI61BtA4tPM3H+M6fpzS2Kn5d1wZ4s0R6Yg3ps1zrn88sKbuxn6B
uakU/ZaT+9B+StMhu1P599R9e+HfMBf7M6Z1/x6/8vfaCcfvrhimHbGw8AM8
HzurodamlVA9l9SKGYm8E39Wfh7o+3bR9zUp8VDtAeM9PK93LEb9LvrNxPrd
7h3s+P2gufo09gfNvMHYH7On1O5C3gQ10fu6hAfp+599dbud4mXVGryX109O
g1qlaYg3tVb011i+FvWLkbnuPcf4oQmoc7TczBL41ae8jfYubEYdZMdL9N9K
ulLumEeVEv7aJw3Yv9AqjuF7J+1kNr6HlDPG4jqMwi2O7x9leBjqc/nsXwrR
zkMZWK+ITXH4bkGuuxl2u6hgm2P9FV9B+5O5Ga/CLy1/La7/6TgL7R4dkgf9
+e2Fjvf/hiP5aKc6cwPiDqch3qok1T5Q9ra6fMfr/2MI7Gb646uga59CP8RK
pa+QXU8NcYy3oyLQfxlxK/qrX1yN6zdbSMUgslvDyc+LRRp9n7W5gK7vWjH2
ne02Un29si/UHOPNSS8XoL3bgtbhPPdlo84VkUpDya49Tn5e8UV7Ma+JIyV0
HfMiEKfPVlqt7Pl7v5bvj+3MyaivjcrfQeVCg/53qbKbys8rPmM62ZdNgYp/
J5O/UqHshsvPA3PzRNj1gicobtgkqKVUU3ZD+XliTaJ2tDUzyG/wWIpXKleT
/b9+nlSNo7ikROrH8mjyU2oru8vPEz2L+ikTplL/hya7qaXsLj9PxOIJdJ9G
Dyf/evKzlerKbig/T4ygH5D/rPup3bq76TxK9dlkd/l5IpMD6PigYeT/dGuL
myq7pfy8GDiA4nZdg79ZWOOmsozs9oABzr9fcAj1e+kRam/yRahlkBrKLpWf
V/zKm6n/i3rQfQjcSPvXSl12l5/X9RfXUn9HBJGm5zd/UQ1ld/l5cXg39TPu
KlRMvNTspsquK7+vG1ZydCuurzUGaryUCLWXJ7rZpfLzRDxKx626OKj4a6Kb
Spdd+Xm1HzaG4i+Q2s1J9L9SU9kN5eeJ3DSa/Fb9CqpfHeumUtldfp4YYaOo
vfN0XJyiONOlym4qP6/49Fiy19NxI3gctafUOkJ24fLzJPoRsn9CavZKIH+l
9iWy6y4/D8SY4dTevJF0PJfut67UUHZd+Xndv4/6UbslQ+g8NcJNDWXXm/o5
/377mrA+sss0Os/hSDfVd5FdKj+v+zdtB+xiViPUejKM/JUayq4pP0/M2LO0
PuvZg/o392G6b3NINWV3+XnFLzlK7VZegMqHg1vdVNkN5ed1/3oG0O/8a2pH
Bgx2V2W3lB/DMEyX+A+6BVpE
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 50->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1wFdUVx3dIQpGhDjJRKihzqSjOSIKhxkBRWCJq0YiCTWoF4hoM0EFq
rEQEpF0LBpKQGYKNIsR0ISWBKJE0kQRom5tCCYEASRBMyQdr+BADhYAJVhli
ff9zn8N7bwnkpZ0ic34zmf+8s+fsvbvZc+7H7sC4lybGB2iaJr796/3tXw+N
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRiGYRjme4HeXZSO/n93gmG+p8iNDX9z5Y8sHbqlozzSf6lv9yfPrGVb664m
zs5K+tif85vljRtdcZb5ZC3XAYbpHPLrj4648kaseueMY/70f+poh3m1vDUX
xy+GnnTyM5Y1H+4oXiYHI+/tB+oc46+Evi+6GnElr/zLn3hjwxyqH68FlHP9
YK53xN2j5KXPuZhTdMz121xyy2nH/P3TyF0d5YUdcQz1QQYHnfIr/7o1Yl6h
PzjJr3j77ljMW8S5AY79vxIyaFY2+j+zbyXnP3O9YUy+QPPvpRf3IM/6D/n7
pc+5Gd12AvYPKhzHT3nq5kMd5n/yVMRbt336uV/5u+P8WrS/c+Jxv/IvY0ge
4hrm1nQlf+Wu4x9y/jPXG6LubczrrSntyFPjZFQzxvuADU2Yd+fEIO+MvCLH
8Vf03n0Q+TmpZi7i4lYlXupnvbIC46f56IvHupR/IU0NXYk3p69v4vxlGE/0
Ibk28qI9GHVArImjPC+fh/myGd2CdbdVu7jFcfyvnboNx9MHYZ/d7PZmbanL
3mN3tku10OFJLjV7jvJr/a6Jf9/uijeSp/o1f3djLH62S/Wjq5jVwzZz/WGu
NYx1SZuQv4fPIf/tcbOQ52bqS2cxrueVUd6/fKNj/tv9duzD+jgxBuO7MfMJ
7LfJuM07cb7Km7A+MOZFnPVr/77F/jP6seSIX+sHfWf9H9GPCx/7N/5Xj8H8
xWgVHa5zrtiPaWV2V+LF8MZtXD+Y/zbWc5mrkf+LfoX8MAd+hTy3o7tjvW+H
fYm8tWtudtz/F4E59F5u0WnyM5IxfzDPVEP1RybS+uJU9y+c4s1Dy9+FX9Bo
1BHjy0FVHuuHwNz9mJdkhfhVP8Rt0w9j/lHQ7tf8w7ixeo8r3oqZ2+zX+83n
iiXmQ8FpJ7q0/oldxe8vmf8ZsjoDz5f5w3/UIw9X9IXaQ4s+xXiekFjl9PzJ
lJ3Y/7en3Yn8speUo26IN9Zhvi6f/sE55H/C/M86en6NtmiMj3rLTxGnvzUI
4739ehbqkVggryr/9QPjUY/kZ7EYr0VUT+pHaWLn6kfLX5LR/pileH+hn9/T
qfWHbKyrQHy3arp/aTWdqj/G2NVVVA+noI7KecWdqh/m8Lx81L1vNvwT9z8i
cAXXD+ZK2CvHZOJ5iy+j94Bl91zddz0hCzCOi6BiPPdWeDvy0Bx8fyOe48lZ
ZVdzHuO9Y7vRfo9etN6Y3kzzipz7jnTq+b+jneYdoVFtiB9W57h+uRx2ZCry
XWQ0U/zeSufvHy7X/q29UL+svS+j/on3RnZq/qCLdBrvk6agbspb13/i1/yj
94wdnPfM5TA2ZWJ+raeU0jx7Yzb9jr8JqqVuRx6LXZv2Oz1HVvmHGKf0d2dD
zTmDMI/XHo/A+CtHDcW+m1ibttfxOXxL4rsa2dfEfoFWVYznVayYQfWjlvLe
6lXj/P3f/fO3Ub0oQZ0y8ytRP/SGFPpuqWEWvX+MzHZef5cMQ10yEt7E/qV+
129Q94zIARg3tSGpeP8hli12/E7RHLpwK+ISj5agTmzphe8N9KhivBcxx5bS
PGrK75zfP86s+Aj+MaKA7tvWXPp/COzLyKSf4LsD/cdpWx3XXw0h+D5JG/5k
DvRH/Vei3frsNdDChHxcz4klBU7xRmny+zge9DDWgXpGaRra7S5JExph106T
nzdW+Jz1iCsKhL9xpGI5rvcgqbmF7OYI8vNp/6Fw9N8MjYO/9vsF7+D3IlJr
ANnNseTnjZkQgX7J7Q98gHZnX6T7mUhqFJFdxkc49l//ug/1P2vh22in2+eI
05QamWTX2vo49l9OP7MU8SUF1E7xLYX4Xag0X9knkN+1hp0Zj/FJT32BNPBZ
qKbUbTeVnw8ZsbAbX8VB7WXPk186qaXs1h9iHeNFnWo/eSad5+QE8qskNd5Q
9m+c27f6Tib7fZPIL+8Jau/XSkPoOuRrk537/1c6rwh/kfzWzSCNJdVfV+1P
neYcP47OK74YR+0Nfobuw+1KV5Ldlpdpf3MYtVcVTsefH033O5pUNpHd1IY5
xstTd9LxhY9RP3473kO1NWR3+/nEp0dSfEWo6n+shwq3Xfl5Y6cfoHVVbgy1
s3Es+RWQWjnK7vbzQszaQ/O72HyomdNIv9eSSmU3lJ9P//vsI7/ggXQd+n7a
t1bqttvKzxvr1XbYrYAL1P6hk56q7G4/n/6vKqT2drWRnl3tocJtV37XGmLE
w624P6t/RpoyAWoo1daQ3e3njYyk4zI4qpXqxNP0W6mt7G4/b/QXniJ79ERq
NyQGqiuVPye75vbz7r85HnZrNqk+Qp1HqZZIdrefN9Zx6p99VPXzoLruT0hN
t135+cSvfYyOHyAVYXTdpluV3VR+PqQ8QvYwFd9T9dutyv6dnzf1Ol1/9kPU
zgDyt5Tqyu728+n/L+4l//cj6PjjD3rod/Zn7nWMN8/fQ/ev4S5qv1+Yh4pG
skvl5xPf1I/sI++g/9u0Gyheqa7smtvPC/2GwdTPkFbsL1sHC6GmUl3Z3X7e
2NX1dLxHFu1Pn57voYayS+Xn036A6qe5nc7z6gYPNZRdKD+GYZgu8R9JQiuO

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 51->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1QFdcVx5f6ETqimAyiVtB1kmr8KFGiNSQybBxHNDR0AD9ijWFlauMr
IszExqij3Gg1Jo2GhBjsGHVTbBHjRK0hShS4iUaKYRTEKGIYF8UvJCagGESt
Dee/dALcIL7XaVrm/Gbe/GfPnnPvefv23Hv37hsYlxg9u5Omafp3n57ffbw1
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmGY/w+6aHvDfuwcGKaDI+L+ud+dOjPqtn3Rnjjpl37QozreMKiQxwGG
uUcW9KhsrBuxL+CKqn70iPDK9tSVHfnBBaXfoMiKtuLtLRuLGs9bt56/6E79
2t3XHaW427sue1L/ZlbXT3j8YDo6cvtL2d+/z/XUKNT/yANfq+5/4923DrVV
F7Ks9GzjeXPgceX4cTfsoT1oXWGlDVL2fzfEhAqqW31YhVv96/k7/tYYZ/df
VMz1z3Q46uySxvtaal8p69gcGYJ5e0D5V6rz1o2JJ9us/169KV4fvcmt+pVB
YR9RfkeXuTX/y36FOTT+BC8u96R+hVjL8z/T4ZB3EqiuzRsDMc9nh5+j+W6v
bxnV3ZzeNH/rK4cr61cuOXG8rbow76yz8hrbm2bbHq2/17x+ypN4I6+rR/EM
0xERgw+jvusP0/Ox/fZmqFZH86111aeatLBGOf/rr/vnKdcN3SvepfGkQHxG
48sef2X83bD3DVvQOH7I/R9We1K/1uCIY1z/DNMc+7ov1acYs7CG5vvF56qo
/ld1x/PyypBvaHxYeEw5/1v9Aui52Jr+Z1o3aH7eu5v5Bb5WgPXFjjbrV9RG
ZKjOiw19d1H7V9a5t38Q1LCOvtdzDe7t/1fN2kL5j7itHOfai/nbxDbXSQzz
YyDDH6W6s+8/ResAWVaAevcKIrVr+9VS/X17/hvl/t8DnWh/3vDOpfld77yG
xgGj6He0frAfzMB4MvHVGmX8b57cRudX/ySV+lntSs773nnROTej8Vj02F7l
Uf0Mv9Gu9xQ/hBEfcsaj+J2pZz1av5yO/wePH8x/nPVX/0J1W7KGns/tvHrs
A2SW0r6dvcRFdSxHvq1+fl+e/jnFn8vE+7XLo2idIAPX03hg+Y27Su0kZqnf
HxyaYtH8mj+e3gNaH4RQf2ZawpeUx32rqO6Ny5eV40er9ia8Jyn+iYatpBkP
Yt/v2GNuPX9YXo99jvVDrFvrDzkii96nGBGPuDV+GRcLaX/V/HuKW+OHGfLH
Ih43mLthli5NpzruewX73G8EF6P+69tcN5tRVTvp/kzyLcB+YXI53hf2p/vV
vpaL9wdP+7U5f8n7Kmgc0J/xp3Wy+GQe1bt4cwmNH/rBfcr1R0tE8pF48g/N
x3rktZ7XKL/hw67dSx0YwWspb+tPL1Aexkc3a+8l3qz7gtYLuo+L8hf+W9qV
fxMis5L2Xy1zMtZjN4e6tX8q+iTxewvmB5Hpgt7/6TGfkWqvzofWHiol9elJ
/8+TCQUlyn2+ymKaX0xvF+aZ+ql4HxiznepfxA2hOjK6nlL+f0++lZJP9s3z
aHzQHy4iFdMX0TrErl5B6wYzt1j5PyHjZAz9P0CfnXKA/FO9MF6ND7xEWraD
5m1zZ/glVbwd5E31YW4cTc/39tfltH6QG17Bev8Pj1Sj3ZLzyvy9+tH/i+X1
VXvIz7r5MeU/uIbqVbxTTusiPfWOMn975dgsqvPCjB3Uzv5p71M7lS66XsYZ
P4oT7x/OV8bH+m4n+8CAv1I/DxWnUTvRSdSekTIyl+yvFH+oihcLEmh/Q89Z
vZH8A19aRtcj9UVSY9/pzaRx87Yox5HusbR+1IJ2TqE8k9a/THr6HVLtV9lk
t30cvxboPxeUt3xjwHzK47mMtdSfCZUxsMtB8GuVf2EQ7H0fz6S8L4XS76BV
QaU37NqnQcp4fe9R+n6ibP9q0uUH8Xssg2onYbez4NeKsKwUyreLwO9mdqH9
Kv1ZqGhIJrscBb//NfQTM2leM56fTqqvgZrLoVoEVB6fqZz/xDYX4hMSEZcY
DX0ZqqfBrmW5lPF6n1j4bYWKI5Ox3/CzKTj+6QpS+8lYZbw9Yw78g+einV44
1notRdwM3zqy91+qnP8tczb6KY6B/5g4fJ+ZiNcDXqB4e2G6Ml4mj0L/w8fi
Os19Ft8/chbaq8iHrg9R598wBP27piLumPM7vAcVB27heO5Q9fWPH49+3xyD
fk5Gwa8IKrbCLlLHq/sPnoTzF57A99Z+iePro6GnYRePTlLH7+nh5O2P+JxO
yOMIVHfsTX4t0VfW0/rK2go1z2RjH7oCKhx7k1+r+F3VsC/vjP5dt3DsqNFk
b/JrgVG6Av2G3UYeL8pmKhx7k19L5Mbd8Pcph19RcxWOXdu0u13Pr/9t7I/D
cV//+ilo/xhSEQjVHfu//VpgDHDi5jxNanghTrsQjXZmwW6FPqWOD4GfPO/4
e01Dv99OhVbCbs+IVsbbmVE4XwIV3RBnD5mCdnNgt05FqeNXRCJvHap1m4zn
hUXOdbiB/LWMSPXzwwMR6C8HqochTp5Fvka2c35RhDq+90T0d2ISzv8eeYt4
qFnl2EMnqsevX8BuD8b1NbrF4thRMQR2PVodr619HOc34bqL8vnNdQ/spuPX
6vr1Hofre/8z0IDFzdR27Hqfccp4MeJh2CfgusvroxHnqOHYtSa/FkifUPjb
A6BJBfS8JR21HHuTX6v8z9TA7+KX0HlpzVQ4dun4tcr/Iaf9sefpvFn5KeIc
tR17kx/DMIxH/AvNwQ2E
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 52->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmgtQVmUax78WicgczbxkTXRQHFczLyGJo66ni6Qg4eqKipIHCqwsWadd
dQzj1TbLcAYW0fASvau1WaumyaJuQS+K4QUUCbwsiAc3KWq9cDN3RN2+5/8x
s8QrCx87u477/Ga+eeY853ney/m+/3v7jm903KQYD4fDYfz46fLj5w4HwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzAMIfrX7R3jRp5cfc9Rd/LaiphUdPC/UQ/D3ErIpzZWOHVj9PtllU4/
5u7Uv7VGV+KVQm2cMSCvoqV8a+6F/c771uUXzrqjX9vL55gzTxVnaNvfWlT4
gr/w+MH8v2HkP0a6tVI6XdDq91REYUu6UAdySd/m9VEX3VofHDyxh/LfCTvv
Tr546q+ULy9d/bs7+caKoo8ob8qGItY/c6th7gqldbG6L7NAq+9Nw2nelQ22
Vv8O/x4nW9KFWDWa5l1rQKlb+pNFxdlU/+iM793Kj3s2h/rXq69b64dGrKV+
h1n/zK2GHVNMuhTdtmAdXuNtk96Sy7FuXtAH8/f8Zdr51wwYf7xV6/+iw3Z7
9CMHvnuqPfnmSwO/Yv0yTFNUSQLNi1Zs6bc0DhhDK+n680vkN6tiaXywgjy0
86/o4P0ZjRdz1yz51/vGdp/dVN75cSVUTs++bq3fG7Huntiu/bux+9Vi1j/D
NEVWfUfn77bwpP257V9COpMBb5HuZX0o1v09B+vX/z9EHCJ9e48knYsXx2d+
4bQxoW/Tte/QXCr3cIfvWtKffeyVJN19KR7IoPLDp7q1f2jE+GNB3v9S/+rh
h4/w+MPcbNjdumaRTlMvk+6Nl9bSOGAcSaf52spIrib9/iFdf34XM4B0ZSb2
Rd7td9J5gNxTRfsClbLiNNYT0/Tz/68Xv9saXRjx+7526/wuJ24xtV8Ob/F/
hhuhen82j/rhf/5Ye/Rrf3OlvF37l5+v28PjB/Ofxup6dTP9rn7nVUbjwNQN
50ivAb1Iz1atRbqVB9d8o93/j1qeT3HfX6P9gdr2OfYLi5JwrhBwtob019VD
O36o3h2XUtyaWVS/uWAJjReyw7RNznWEPXJRKfn9OtW4df6/KHK1sxzr+k63
9h+mb94uZ75c2b1N/1+IzseTaTzceHeJM9/sdKXF9c+NUHt/sZjq/zbArfHD
/rIym8cN5t8hFqbT+zui2p/+zxMdV5Lu7FF76dzMyO2Vqd3/h1bi/7lrxdDv
ZJPGCRFSXIX9xBnaN6jAotKWfofyzrpdlB+RcIL0H1NXS/ELX62j8cQ3tlX6
N/MTs516USuH0HmGfb6CyjHTx9e2Sb9mB1ovyLqppHu5XbVJ/0ZO6pf03K6s
Rl6Bo64t+eaI9+j/BnFvMZ5f9AdunX+q8hLF+mduhNktEefiI/tD50WPQ/eh
d9E8rO7tjX39+3Xa9/TsN3ZhvLgrn3RrRz9yhsqp3krrAav+N9DtqDTt+4Gi
Moj2D/JoKelFRmyjczr1dCjlm2eDSDd2X7tSu37wHkjl2nNn45zh5Rl0HqHu
e510I7z2k/5UeOo5bf7qI6QPI/IQzZPKL2ofxaWH0nsPonwB9L+hTHv+YMa9
R+8FGdd60/glQzZSedbQXnR+Knr407pDTMvSzv9iXDGdb8j5I7dR/UF/3k79
3uxF7xuoRb/Fucz1DO17FnJrz62UX9uwkfJ6RK+iuAuFn5J//1g8F8/yHdr6
07p/SPWkfbyO7r/94kK6LvmKrHngdXr/Qe588ENt/8VJSfG5/Wc4x12RWRNL
cQcqyJoPHIpw+h2epVL7+/nTCxvoeU18Poqe00NmKl0PgTWKZ8O/FXHN6n/+
LPXbsWIwtVOsK6BzZ2s9rIiH3xHpivsJ6ocz5BfZg1Lo/mvJNM+peFjjC/hF
zRltvv3kJ7TOEycexDp6lTd9nyIZVhXALwMRd7Nh5E2HPstmwW6ZQFbmwBqZ
8JtZ07Xzr1n2HPlFeBTuR4UjL8+HdGs2bMG8+1aMNt+On4PyPeZhn5D3CVnl
/SbakRJYT/71Cdr52wiejfhTM1D+vDSUF7CcrHUynPLFc1Mua9t/OhD1lE9C
fHAG+lO9Dv7abfX4HV+s19Z/cgTiKp5FO/qhn8ajaL/w/z3lWeEN2nyVbyJ+
znjEF+B5yuWu52E1wL/qff36Z1AI8t8Zh7jbZyJuP743e2wU1j93hGnz1e4+
iBvzGOwWH/QjCVaOhd/c10ebL+f8DPc/HYTn17k78u6HNbPhb4z7KWLeP3C+
1LUz2t/g28Q67oFfuuKa5Y/JJb9Ut6G+tU/QteWyjhz4G+Oatd8ji/zm4DDc
f3MyWbFschO/csU1q79jKeLPnUI7Ds5vYi2XX7nibjZkTgjpVMx8Gnb5FLLy
A1ixDH5rT4h2/apOT0BcCuIMORk2AVYmvYz7X0/Q5tsBvyK/egPW9puG/LXT
MX48lIj7x6O1+VYh6rH7wZod5yC/eibsibRL0E+SNt8oC0P/dkxE/uY4XL9m
od1D6ihfrUnW5pvPuJ7f+lDkDZuBeuPRXyMjHXlXI7X5Mhr5pifylU8E4hTi
7dij8F8J0z//wmDU44f7MisGdofLDi9DOReD9e3PwPpKxOH5Gk8uRnuCYWUi
/DI/SF//M+inlYDvTQUuhR0Bay+BX8zSf/9mVADK93P9vjahnWpnsOv3AL9t
BWjzRdow5KeMRvuHPYo4lxUuf2Ncs/zTnsjb1wP99urS1Lr8liuuWfsf74ny
IxFndLkfeS7b6G+MYxiGaRf/BBsr/dM=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 53->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1QFscZxy8lHY0fkSZqp0Kcw6hNjBPNjBijBi/YGClGDFGRiMmJllo+
KmLxxUp01ShoSKMBq0kNHFSIwWAi+YACxkMTo1ZBIuIX1lMDFgEpWmm1Q23e
/3PMBNx5U15mWps+vxnmmdt7/rt7y/139+5en/CFwT/zUBRF/frP8+u/7grD
MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzD
MAzDMAzDMAwwqyI+m+COLij/gDu6zqJVNv3hP9EOw3yXELGvn3P6Rtt+/SuZ
f7T6zAsufZVWKpznRc8Hz0r12Sku9WruJNN53lJnWe74VziKyqE7UHWxK/7X
PDzf5/mD+X9D7b4VvrGGTrwi9e+Qnl+68oV2PR6+FZmHpPpvQ5/V52Onztj1
iVt65e1znzp1euMHf3FHr029mu3UqfdcdHmdDPO/iBZZvQf+2LdGug+3vMqw
7quvjZb6xyw+ecKVL/SX72uA/wc/3+iWf0aM+Rz+b/6+W/4V9yd+geubdt69
9m2sXn2q2P/Mdw0xrpb82Vxa7YxmwQ74WbxSBd+IM49h/VcfKpKuv9bpFte+
2FSzEc8Puzzc2n/rH51cj37Flhzrkv+mXzrE/mWY9gj/3Mvw5/AVtYiHi/E8
rrfm0r7/JR3zg+Ez6bIr/1jpu3d887y2uUcp5g+PiadRX/oDXVp/xa2Kuq7o
1VlLK9j/DNMebWIx7d97zazHOh+zDn43ayMRtUE9UG49HN8k84/ecATrqpkf
vR/5cY6Ve5wnbiXlwf8x4+E7/fl9f3blP3O7/+Y9knLVW3yI+g8dq++Kf/UJ
KYX/Tf+rV2r28/zD3GmIwDePwLfjB+P5WgsvxzqtD7pB/r/cDcda9rvS52/9
2lsFKG96Eeuz6jO4BPk564qxb9AKaN99tLBB+v4gomQe5pdeFxKhn+xY9s08
devGNJwf27+2K/4xXx7o1vovzgRiXyPCWnd26fuB172n2P/MnYaZ3nsb1tdj
qefh2y9TyP8pgc3wb2O/JtoHFEif/42AHX9E/uqdyNPDkmifkO6F94Z6vzV4
btAXbJfOH+LKqlfRfuMC1CPULcXt8vyewfqvH1Dcev9nvpNZ7txXaEdD3ft+
0FbPfd9zr/3H4+uc7RsR1Z16ftGrVszHuPf3O+HU61G+J91p31hb/yrPO8y3
Yb4wuww+HFeC73X6pfFYb8VSL7wP0BLWS79/mQHX8H1NWf1oJfSRb8In5qa7
af/wz/7XsH7n7a5xdR8ax/diH2FmJML/oj6M5o38IOiVpxOuduY+tgqG03vL
vffgucNIrO2UXiR8iPea1rFYzIdm3MFOzR9iyuIY9H/Zk9dxPZ7r/toZvZb9
eBH6v2QhjeegQWc6o1dHDse+yjBTd6F9/WAwzwNMR9TmGrxXN/r4wr/62xHY
B5hjUuF7MWIIvgvoiU9Vyu4fNfw45gW1Ngf5Rqs/1jlr3yjy7YG0FuhzfA9L
77+AzfjuaL02E88JYkomfidkLo+i55GQfPjHaiiUvj9Uj7TsRf3PeOI7oXnX
moPIF3Owf9HfP07+WXu3dP22SlPx+yIlMRjzmPnBJejNV36H/Ys5J5J8u39l
s3T/UzQYPjX7GZi/9F9+gveeqr6Inlem78O8IzJypO1rxWM+gr6bL35fpN66
id87iOUex3EdRQm4fjXZUS7dP3mlvgd9VnIWxj8zD99blBca0R9rkR8990yO
lv7+Ue174/fQ5TT+FucPx/3Eud8w52b5O6OoG4jnH7O12zZp/9c+lfZvzStv
PC3N08aFrW9Xfj4rFtdfSbENc0TIepneSt60GfdJ/OIM6AJO5ON67KjOoHJl
CeXdpg/22Ypy76ZnMV7jv8iFfixFJZLKlZ/aeR3pfTMZ4/PjJPxORKurxnhZ
5ygKb7v87zeS78j5d1EY3Z8b5iJqT8xANB0/p/KY0/CxMXq+dP00HBEoVzfG
kM5nIcV7W2jdf8gH/lFXz70m3T/0J51RRfWrD1RQP05NwH2vXYik+Hlci/T+
/Wwa9XNLOOmi46mfUS/RsRIFnVZW/Tdp/+s00kfGIuqjllD/31hDsagW/da/
apT236gLofYcoRQvxlHsbfcjwUHj8HqFtH2rZBL10/tFij3t8QuifigbhkKv
xh+Rjr95dCqV96F+WGcD6DpCA6l8qw69Vj5brg8aRuM/kfSaRf8Pa+oiKp8e
ReXNI6V6UULtGU/+inRbVlL7ZRSVaCoX/wiQ6vVCT6pfnUcx9DmKIRSNTVRu
2Hm36aNG0fkaP7p/SgZQ3D2gXblh593W/z/9iPKX9qV+Vl6hdcOOil2u23kd
UXt0p/5WUjR7daP27NhWrtt5dxpGxrPwp+kfhGj5zkTUEk7RujP/F4h67FTp
/lXPI53ieI502+ZTfY/8mmLZJUSx+MHr0vn/rhCc16NTaZ6YG0b1xNGx8d5V
+NcYvku+fx4wnfr96XJq5wcrKW8OHRuzx96g/+dvpO2LXOq3KHyL+ns+g3Sl
SXQsskn/wxSp3np0GuWdCybd9mUUHfOo3tTZ8L2YtUfaf5FC46qNtutJC6e8
pFAaj8BVaNcISZNf/8Ok18Om2XEOjUfZAqp3QyLtn0ZmyfXFk0kXTveBOoSu
Q1sxg+qpGEvzb2ugVK/m+dv9fYL63zSFjq9S1NKpXN/pL9VruY9Ru6OG0Lh5
DqX8YRTVCVSu2HkdMQd6U97F7tT+sL6U9whFva3czuuI8LtJ69tZe37fe6td
tOzytrzb2j+h0HWmNuC8ucpe9+yotpXbeQzDMF3iXwVh794=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 54->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmgtQVdUax7cPCqyppCuNkcO2qMhoJrm3FIvrLou0DA00rUB3hgpciUBS
I6ltPhofjfbAKVLaFJKKFHp8jKmXBViNhGZ0CUPTFYqoiMhDuGpZnf93mMlm
RXnONBl9vxnmm732/q8X+//ttfY+fSckRU7spmma/tPfVT/9eWsMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzBEUcj2wW7IRNSKre7oGIb547FrTlc5/Sl80g6ofCruqfn29/jX2jyxXHWd
XTnwYEd6q7XiPed5vTyywp08YQ99shT9r92+35M8Y64en8V5ivm7IUISyZ81
sSeU/k/y/V9HvhBz/JA/zOW1Sv1vcihtFfQDx5x0Ry9DXsD6wgx6q8Etfd+V
70Pnf7syfzFMZ8Co3y2U9/fyRYfw/D1xmdJ/hj57T4f+d2jHneftoD5u+Vff
pe1A+zfHu+f/SWll0BckuKf3OzcH/e/21D72P9PZkLtehT9ljC98rId77UYs
jFrtjJbxIp7/smCD8vkpbsrt0P9S9lmA9ffTfevc8Y8x6cAKPL/HTu2wnV/l
5nnzoL9W+8oT/xqpjpfY/0xnwxwUCf9bc7/Hc96YFYj9vGhOp/3ybSPouend
cEH+lRkvYt1tnD34JXQzFh/1xD96fKtHehHX8An7l2HORyy4tRL+9/5nPXw/
7Y2jtF6fBb8b89YhytyJyv277Lem+Ofl5qhzOYVO/bDnn8E6ooeJ93amX/ox
d/xnicaV6N+yrvV/Zf8ap9aU/JX7z3ROZFwyns+2yMBz3r7Oh3w6vQ7rAvHv
y5EPzPEzlPtna2j+EqffjZmPnL8+mLw6FXmj55Es53nzjMM9/145Lh/+XxpW
7dH6oXt80Z/pP3l6TBn7n7nYMIdEvAmfb8+Gv/QBDuzzxcvz4XcxKrAJ5UWH
lf6XCw/T+7XCdMoX23YdxvFch0TsvrEW5cMGq9+/VecMcuYHbdGzxR35w34/
4LhH+/e336rxRG8fLD7kiV569fcof1l+P2zk/MH8URgvP4B1gH6ncQR+nboB
63153WfwrxW6Wfmdz66I/xjrg9oJ32D9cNXIZkS/b2sor+Q14vwjBR36Vxhh
63H9LXdtceYDkZgAvf1cKO1D9s5svqD3D4uD8B5TxH2BPGQMmNp4Qf4ZGIzx
igVzKW/UvOPW9wPrnXLkDfnQoQvSi33ZDuc8WNHHsW6y18d/4Vb751KKsD7b
tP+VQjf0TOfGrEiF780DZ+n79n8C4H8jdx18b3gdwft/KyVE+f5dNFdDpz/e
Sj71fgbrBRkQ2QLdnvBTOO4+WPn7HXnqEvq+t/NzRPFpAn5nJKKGwu9iVjzq
kY3/UPrXCCvEukGGr8fvD43itVTf8Exav5ydivZF09Im5f4lNhrfPe2u+dsw
D23zd+K6K7bCt3ZcEtq3x3dR5h8hczej/sI2PJ+Fvhr9se7/EPNn5nz9fxw/
NkypN3uXOqAr9v0A7VRGoT6r27t4LyOeDUb7xqUPVSrbX1ieB93YCHyvERlH
30VcmIJxia+D8d1D2G1bVHo98+p1OL9sEPKv5uWDaH46fxPqTQvGvsncf3SD
cv5Hp0JvFTtwXrQkQWceSUEUbdfTeOalrFPql2RhfyfPjIfePHwT5kPG3ELx
wcdRLkfk5ivHf2sU6W/7rADztL9pK913FMVHZSgXd0Qp9drwEvyuy2jMyEC8
bDDeN9k+rthlKcqte0uUv/8yzfDFKO9fi9+JyAWVa+i+pShvpHJruOu6iwzp
E0t+/W8CRS2FfFJSj/tO941uxTxMn6/0j+idiHKjIA/R3FGEaO0txf2uTwts
QywJaFGOv2w06TdH0z4jdSX1Y2M19GazhH/tnLWtSn046cyZ1H+xIpv0ry+h
/PFaD7Sv9fRX6u2sKaSLoHnQJq4nfZUX9EZsKPTi4ZHq/mfHUP/vMqme0bMQ
7VU51K8e+9B/66u1av3pSLoudDLpq56geXgpk+r1T6f8ueQSpd7cNIr017jm
b/b9NP/PU73ajiD6P9z9kfL/Z943kvq7qBn5Us8sO0njL0U0H2ui90LTYtT6
cd7Iy9JX4DojsJx0r+yk/Fu7m8q/9FavvxJpX2lN30PXV22lddJeinIalcsp
6v2nFafTePtXUD2+FdTO1RSN9nLXdb9ED2yl9em1XlRPz1LS+5aeX35Dq7L/
prdG9b75Hc2Dow5RuGJ7ud1+3UWGOSmihfIk3d/Glmg6Dh6C+96+/QB8Y/3r
E+X9Jx4cRc/nTY/S+Zh0Op67iuIWun+t5fFK/1kjYqjd0BMt5JdttG5Y20B5
x7/vaZzPazil0stm6q+xfRHpQ5IpbyWXUn/q0s9gPAFx6vyR+AT1Mzmb2g3K
o/H3yqfyzIOUP67spu5/Go1bL6N+6MWzSTdkIdVjD8XzX96RoOy/delo6mfR
GBpH8gzSf0P9EgPmkM7/hHr+X6d1lvkd/R+0sxOoniqaB+tkFvR64vfq/HFs
LLV3DbUvlj5J+ntj6foPe2Hc5guxSr1xbDJdX0/9tfuNoxhG9YgzU1zzmKjU
S5fe/pzmQfSj8Ygwilo1let1k9X6FXT/Wkl30jxE9qf22mMClRu5EUq93rUP
lS/7gZ43dZRnZXt8m8rt9ut+qe/di/pf3IX6Ud2Tjl3RcpW3X8cwDOMRPwJz
FO2I
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 55->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1QFVUUx9d0UkdHcTSDxFwzLamwrCQr81qRfViGjo6jERuJRpFoYJjJ
tKkRmJqolZ+4KPmVlRkBpsClFEVEM4EETJdQSgKVD4E0m3z/85xRu6E+bTI7
v5k3Z+5997/n7r79n727+zoHhg4Kaqxpmn7q43bq00xjGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIYBVpm+
qe+/PQmGYS4r5m0jfnD42ohOzlP52/LOKb4Q3xsPjflWNU7ErjnQkF68E/eR
43v7x07SpfqSv3WzQ2f9sWkX1yeGuTisPyJL4D9378NK/3t55zfoK+vx3dBf
+0G5K/4zYyrDHDo9qfCoK3o7NjkV+fded8QVvRHrv8ahE+sX53D9YP5v2NWZ
uD4btW2V/rM+8dnTkC/MKfHwvVGTpawf58OaOXMb/Jcc4pL/jeDn4VvT3TW9
HT13HupHZD+b/c9cbejf5B7C+jg8Fddpq7XvVpznIdcuQzvzY5z35pLXlP4R
ubKwIV/oSbOxfrey+7jkP6l5rUD9eHfGPpfqh3+r1dBNCL6g+5S/wxg75yv2
P3O1IT3yK+DPg3vhD31XMcXg8UWOKFfNh29F0ZSLu37nen6J7SyV38G/A31K
L8U/dv1Pv16K3ryvaTr7l2HORiYEYP1uib5l8LkcDZ+Z3fph3S7iXqc4JbJC
ef3PyE87u7+nT7qjP3T5bNSP9JSdqAMlvZX68zKkaBHmc72bS/cPVwrCfCTj
vzx/5urEXp21F/4/sIB8P7IWftd39qXYZjzuD0RCUIPrd+tONzwnELe/ufDM
cVbyJ4mOeiBfcHfJ/2bE9pmYX9rnl7R+/7fRBx3J+i/Pn7m6Ec9GwV9G7A6s
A/SJb8Dvev6MSjz/Wpmpfv7n3ykb/mzXCfVDvNgDdcDesiAX1//vux7Edjw1
l/xvNR64DvqAQpfW/+KhtYsxr843ufT8QJuaiecgZt7vrj1/6JUejeOTe7Nr
+Z0Y0/yXcv1g/inspvfSe7oTbeg9WVpYFepBzAtoW3OXKJ9/i/quuK5Zofn4
3oibB5/qBTvRFos6Uh2xPqhs8DnhyoXRZ35vvNZ8P/SBOp4byFt8Luj5oTzQ
P4LyVs9xrDu06e9gv7Swpy/q/aOZsW0HfN97LPKLNYcv6v7DWjhkPfQH87C+
0mJmufT+US8/jrohW8VsZv8zlxvrwSfofz21ven5/4lJ8Jmtt8f5LsPrf4Z/
S2/eq/T/9mnQi1Ft4W8jLKQGumcSqH5EdTkGXcTJEpVehi6Az2RuBK0jjppU
Z6LKqzGP2UnYnoj0UNePY37fwGcR3eAP2aEH3l/IHn4YL/t3qcM8Og45psy/
uzv9r2hga/xPQLzdF/8TslrEwfd6yGeUf55vlVIf0w0+lzGBSRiX0hr3+XbB
jbhvMgcnwveiIrFaqf9tAJ6TakY1/c9gZMsNyDsprgD95huYt9XKTVl/ZUU4
3m/IXj+vxPelG+LRTmyG+dg+67AdY3LbdSq9uW/qF+hv0SIRugHjMR+r54PJ
6M+LwPG1O/RJUumNWaHYrumzJwX6eYMxf62sN8WExI3Yr2veVOYX7X2x33LF
k8hnlUSsRTsqEtGuHYx+ceegNcr9v7t4Ffqje+I4iHsOYj/EyxTl8F7oN+c6
x52rj1qF91Oy8cmJ+N670Swc/3spak9Rv5hE485Fv8vPhH5POI67eG++Bf0U
imKHs9+Txl1xbB1L5/VhinZpIqLZ/55atINzEfW3HlOev2ZlAMbLpZmIVnwh
/GIsSYXO/LwA/rNHB9So9FbsKNI9MpXyG02QR4zuiPFmZFU9tp+wuFZ5/mSG
UJ1ZHoioD/emeSYMRLTym/+G/rnldcr5p9I6x/5sAul/Ir3wqqd5TTiAvFaX
HGX90DOCSVcXTsfh2yG0vf7dqH696Hkc89syVKm3DxkYL4JG03FvNIti50mU
f1wJ1Z+TRcrjp6UKyuvVlPLWNqLttbyF9BsT6Pd4tkpZv7TKrVRXj5cjmtHR
VMdTaL2ml7XHfhjHS5T1V48Pov6a5jTvwhyquw9nUSzWqd/zJXX9LnwU/aKo
Gc3/7u/QtgJ30XxWt6X9S/FV6uWq/TTf7B20nQw32m9nNCqd/c5xf9GnVtD3
lR6UJ2bAWfF0v+Ycdy7Gcpq3MfEGynvr/XQ+dKcoTvc7x11pyEV+OD909+cQ
7T4mnS9Pfo3zXrydDd9I9wDl+avpM8ina1dQjMqj87WdifFizO3wr/HwK0r/
2isDSTejlKJnS+hkjYm8ZlwW+XfLdUr/2o2c+rRFlLd1NLW7z6f5euyGXjtS
rMyv1VFdMg4tI3/ZX1AcupGOxxgLefX9Ncr9F+OGU163sZS318e0vfJ0an84
qo7qR67a/w+QXpa8SnkL6Dha9ru03diOpPfwVeePHUH6o74UYwaTfv0wmsfU
ePodOjdW/345d1Debb/A5+aw7VS3PtpF7TABnREYrqw/un8TylefivHyfYvq
56fLqP1LNrVfdVPXr5Ht6TiNSKO8oZspTshA1G/cRu1br1fq5dCutN9BZZRn
8mHSOaNpUL8c1lWpN/28qL95Femmt6w5M0pnvzw97hyEj079v7ej/ehwA/1u
nhTtE9SvnR7HMAxzKfwJz3XQVA==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 56->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
qi/dH/4qtRdE1vyj3FaPYRimSfwOZpTFug==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 57->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 58->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmg1QVlUax99Rt2kMQ9M1Ad2u2LJEuoiLo2LIZcnMAhdFdLJFrw47WkuL
qUQTaLdo/cIlbW1lRqsrKQviR8iauqVeBZdMQTHURNFrhvKRCSpfqzUr//86
086c3HjdnVX3+c0wz7yX8z/nuee9/+ecc6HvtMRxv+nocrm06z9dr//c6xIE
QRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAE
QRAEQRAEQQDavNyisP91EoIg/Edxwmbd1NfG1cDPb8n3hVPPSN0QhNsTM9nv
yzZ/WhO7fqXyqf5WyynVdXPZSxParpt1vfa0Rbss8IQ7Prf9yiKgn+BR5Y5e
HzzfRv7dy2+pzljW8/lSp4T/F8xrPqPhu+cHwf/G+qqvVc+/lv+4cv03pobA
/0ZG7cW2qJ3wqW+Pf4yic1Pg2z4D9kPfuOOiO/4zdiRCb2vebuntukXZ0NWP
/0L8L9xtmPnDT2Odzpy4Xeljnx4V8F9sjtI/9slR6nU16NtY+HfaNzg/aBdG
XnDHP9ZKzx3oZ0SnOrf8t8GjuE2n+85tV/25gTGgZGGbznnmxQrxv3C3Yc/Y
Ql/6V2J/bs3uUobP58s/w3MfWYR9v/ZGsNI/esBncWj36t7p3/29MzPDwudX
g3bDf794xj3/3hhnaRe3/KunhC5H/kMPpNzS/r9PnyXif+FuQ/MtOgnfr7x8
huf0ynN4zsMXwW9G3rtY962sw8r9v6vWa/J3r+v351v/8tn5dSk+H45Tvj/4
d5gv9Mf+W3uqvOZO9p9R7ffRnZy/cHfi1PvivZq5Ihr+tD8oR9Tn/KgW17uk
cd09vfKm52fn3lFY580jTRmoF10Wr9h1PWqdQnPQ79ltDW7t/z9NeQ/7ixmx
J+9k/5jrXxH/C7ctltfS8/D9+3Hwub3hK/jV6FuGdd8ZH3nz/XdhHuqI/tDy
cqzXv1uCc4T2WthR1IWEILf87/r45TzkExrv1vv/G2jBh2x39FpqSjTuP2Hu
x7cyvnNwTbH4X7hdsSOHHoffe1RcwvMe74mo5790GddHN6jf/31esw/XtzyC
9+PG9ijuHxINB/6vfoy+v/+tm/rfjpu/nH9nXLgFfn+9J+qR1eyFuqPHT/xB
9cMqXJiGcZsb90K3I5nnmg9L2/X+33Y2HET7PxbgvYVVHNyu+uW8krwL4w6c
w/erP/Zs1/sLa8wknHuMje/g/GRcObjHnfrhDO/wvtQd4ftwjh3D+mx4xVTC
L42b4XujxAXfO8YevB80HtxUrXqOrEfHcV9edxH+MOd0hN56Opt15GjRFfT7
UYby/Z+RvZj7hKyxhzHOoRjWkaxA1p8IC3m4gh68pNI7v8/GuUPL96DfRxZ8
gs9d9yMfq+dE6HWPMUq9vvE4fKpZpfg7g+HbcATtLkThfvWAtewne5fS/6bf
8G1oN28R6pbVMKIQ8ctvcL9OaiZ8b/V+V6l3xvbZjHHLqnLRvrIW5wQ7p4Lz
GjYJ+VvTks8q5+9Q2p8x70cHr0Hsudri+a0F/WiDj+Pvs9bk6m3K+//g6nqM
b2RuQrudJYhmSQzycuafwPy6ukcVqPRa1Ox1nOdf8j6yErci/6eSEM2c9K3M
IyFPOb5WvhrXVzRv4nPSi/dRFozo8v8t/u/C8nw0S1nHkkrRTp/lh3OiteoN
5GMmLUR0+k3AfNhN+9co83/5xWU/aF3xSFa2sxpSx7edcx3nhRXI44u/p2O8
J64iOsVpuG4vW4B2txvannX06Sfr8JzZwT+DX7XdjzdiPtfXIJoRdqNy/Q9I
pb9+msjn1O8etvs0HlHruLYZ+nk7lXrXmByOf99Z+rNfJ/Rjhs5FHmaHcOi1
KO9m5fi7SqGz++6kflx/7leGr2JcMLYJ+QUHKvVGXhZ0Wmsy603hQM5D7yO8
PrUEedgXfa8o869/Fu3Ma0mMB2ZSd+Q0+zt3GeO7Gr5V6+NjqZs+l+1DWDet
K7m8r6kJ1LV0v6ysH/GTOX/DpnHcc8xHu4/92ZW9WH+rfqLUGzMHcfygEdR9
3ZlxVTdeT9f5/fo8qa6/Hi62f9KX451l3bWT9rHeGU/zvmp6qfUPd+D91vux
3doz1DdVcD3ZNI79Gveo9Q88xPEdf/Yzegnj4j9wHasJp/6ar1Jv9mReTsYQ
/n7jWOqiOa7WbST1Id5KvevDKuZb3Mx1Yuk/83mb0Y7l/Bibq9w7//6XsVsr
6PcFNXzOO/wFPtXzM1uR7+SEFnwPxYbSP855H7Q39oXiObffHkO/e9bxud9d
TH1t51al/tQ79HnpdPbTNALRmT2rifX7PeitYY+0KOfPyoXeKatkvZqnMf9o
P+jNpfHQaZap1Gsxb0Jv/DWE9Sn9EP0S4808nP28n4N/U9Yv60AKx89Mp+5U
Ieex03lef66M87CmVKnXG+lve/WfmEdBA78P1zHGiDDMmzZpilJv7nwO7azq
SYwTXuN8Tkljf1tf57xmhyv12qhfUbd4CPNdx6gFRDKv3s/ye92QodZfGszx
Mh5mnVo7iP2tGsZ+ZuylLiZGWf9s76EcN86H+aZFsF30E4yeIzmvPUYr9WYr
x9GH+DHf1HDG7aHst+Ax5tM5VF1/Bwxk/t082a7558wjMpBxthfvo7m/Ov9c
f44TwPz14iC2K2fUQvqx/23+6vEFQRDawz8Alba5ZA==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 59->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmntQFVUcx69JZVj5aExKqzXJRzVBRaZouJWSIAlKKiraokmTFpdSEVFs
STRRKx/oIKUdw3zkg0c+MkBXyAIhBCWVRFkMUPLBVTGtBqv7/d1m+uNkcZnG
st9nxvnNOXu+55xdzve3Z8+101jrkPHNLRaL8tu/1r/9a2FhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZh
GOZ/ivFqUEXfJujFlWizKXqGYf45lKLq7+z+NGMivpf5VA+LkPpXfTc5AfVJ
D6fYo1rQscgZn5tzl6/dZY8jq485o9cabJ/bdYrbYwebkmeUfclRnKeY/wtq
R6/59vUuRnlUY91nN5yVrX9x8iupr8TPC6aj/t7bbPaodfe3Nco/ZS5hdt9r
1uX58G/biMbpHSi/uO616/QeH0rn/1eI2tPIX+bQeyvZ/8z1hvp1wiH4o84l
Q7a+jY2XynA9e1Wd7Lrp2e64rF7xy/wEvnmgL9775uIYqf6vEG9+mYv84Rl2
wil9wJIC6CfbnMof6pDXVyD/bBnL3ynMdYd5ZecZ+DPMBz4Xrit2w+89C3NQ
f/o4XZ+3XupfPaAhGdcHdU7443UtMeM9+/vbCPE3UL/9m1PO+Md45s54+G/g
Xc69/wO8kNeMrntym+TfXm23sf+Z644+rsfJ7+2wv1XKrVVY56Ofh9+MrTXk
+zORUv+rOUvEH+vVNnGx6C/03Azkk4IyvP+NVqGnnfp+f3b6RvQTuNCp/PE7
SuX+6GvpX+2z1js4fzD/OuI7k/8P+cGfxj0z4HNR/kstYtYryANmxqCr7t/F
vFapuL541EDst/fG0blf52npKN+Uec6p939Izmr083HKkf+yf5SYk5/9l+fP
XN8oHT1wvq/5B8LnessA+F4dWkzxo6NXPT8zl9bhnFBJzMZ5glExvhQxut9R
1N9xf6P8r4XUvA9d3/bIH5bgjJNO/X5Q7L0O82h55zqnvv9d5iH/6PtfTGuK
f3XrnDz2P/NvRelf8S3W+RM74VPT9wN67ydOOA//73jpvPScL70DztfV8svY
RxiTeyJPaGfb4TtCaF7QKR5BV/f/O/0mY7yR3XD+oM5sjv2+OWEz5aOy8L+V
P/S5ITiP0FMv4XcJdVqPGpSjoht3/njDyhLou7wGneHVq1H5Sy1dhu8WtcHE
d5VxW1mjxhePe0OvHCnejzh4ZI5T+St3YSrnHebPEJ5ueF9rm+ro//H4vwe/
6mtsWO+GzY/O3bK6S7/f9byu+F3eOBZBPk8oJv3icESt1dYL8HH3XOn+QTnb
AJ8J39XktyobzvlFmyrqL+oI5Z24XGn+MePXI1+I14O/gL55YT6dX9Ri/srj
hTSf6P5y/669uAvXF+3YifuojkTeEH6TsB9S50wiXeZ0qV43Q7GvV9+v3Yrx
Hum1B+W8JDwv0/sh3L9e0kOqN2suY3+jd70Lv5cYC7/Mgq7DIuybLF0ioDea
pVZLz1+8fNZivBB3+k7K2/wRYnw87kuxuaMfc5wtU3r//ks2YfyGzcgTqlKK
fY7xZhKdm/pZcW4qFnTZItOLpS9uwDguj+I+jAvDcE5q3DoRUU303Y6yVrZR
Ov/yaZi3mNr9U7TvuCWN/p45iHpSPp6raBH4sVSff5h+t+p6AO3Ml1ZiPSjR
OYhayyCMrxa7fCrVb39+PfqfXYL5aqIb1oFlyBREURkNveWd0vXS51eq4PxL
P9oM68DwKcA8tdH3rcF8BsaiXps7XEj11xj90W/hD8M1iNbpgKR6PL9xFRcR
A04hGmkxF6V//z4J5O/ZJYhiww1oZyrWH1DeU3cJulaDf5Dqk4vInw99RfuM
+XtovTcMwDwsY2qpv6pY6fiWxErKCyO2UYwYA71SRb6xlJahHzH8Len4irGI
xu+3zpFn2kKnfeBKer8PKT4zpF46fsoW0j09k/o5vhtRLKK8Zc4spec5sPqC
TG+smUHt3WMotkmnuI/ynzbHneY96nv5/mtKJOXJM+Po77iB8q45wUrlqIcx
vtltuHR8bVYYjefRgXRHT1He3/QdoogMp3XxyjT5+CEPkj74GOniKSreDv1r
bnQ9zVeev5t3ovrW5dTeQtGsO0wxtZ7yZidFrt/enp57RAbtV+OW0ffqHIra
iWyqP+Au1VvybzlH7RPpvFvEIuq3zyTdiaWO918baf7WxjnGr2lAO5FX4hiP
onHqRrqPg27y8a8xQhyidd3sx3rKk+Rbc7YvfKuumngZcZnbJen97yKdmTwK
OuWJbZQvItJQ1itfpX6s6Zel+TcujcafdDP5e0U7RC28iPpxS4HeLO4k1Vv0
TPLXAxPRXm+/F2Xt/p+ovwcLMQ9Dv1k6f8NvFd33BV9qP/gwynqGC8pik+N5
LD8kzz8pCZRf2q+k+0jdTOUrVVQePwV6LfYWaf4x9Riaf/YyRKP8NM3HPYvq
C+4gfdZkef5V36Dnfz6K7tsaS/ozs2n8YUMd+bu/XF80wvH8hlE/XhrF5S/T
fRyYSvreveX6dD/S73uS5l/fn/R3P0fl0NU0nxZjpPlT9xjgmH9P6ucwzcNI
DqbxfUZTfC5Enn+nBlL7yiC6Hj6W+jkYSvWdI+m5NHtBrq/oRe1neVLc/RS1
D/ahdeDZl/o531uqNwu9qf0sun9zxCCKgqIeSM9BvN1HPj7DMExj+BWintf9

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 60->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
C4IgdId/AnadxgE=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 61->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmQ1QFdcZhm9tQWgVxCYYJ1FXrW01+JNUDZN0cJOiOGO1SUdirGPdYG2M
sRCko0KjrnY0UkJEtCaCZlZUxKqoSASJwZWiiBFFjYpcZ1xFUUTkp/iLSsP7
pjPNzKmRy9jp2O+ZYT7u7nnP+c7efb9zdm/PiKhfT/muy+XSvvrr9NWfj0sQ
BEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQ
BEEQBEEQBEEQhP9T9Mi9J4e1pYO4OWfapBcE4ZGhHfCDP43p48+pfGr7Z1So
jmspecl7Wv4xI59vOW9r67P3eJLA8o6jW/Ta4u221AlB+O+g9WhMavGrHhtZ
2eI7Z3bvWpX/rKMTTijrQtd30lv0dvUF6PT4F661xr9WXdg4+F4PT4Xuj6OU
438b2q0Fe6HL+by6LfVDn9a9TOqP8Lhhen1W8KD72g6b44Z/rzYp/WePSD+v
Om40jcjC8WdKv4SPX9xW54l/rKMVf4e+5G2P9MaK3BLo1vSo90TvlPtubtFZ
Z85fEP8LjxtmXGlNy31tRgWdgs/3LVoJv+UcX4P9elMvrJvGmRKl/430cbuw
PzDDk79xvuH9ZdA/6dqPfnM986/ea+cMPD8cL/BI71T0Yn07MdjdpvX/4CxZ
/4XHDqv+5iX4P6gK65sW0e40/J4QB7+bHzSjPtiJr6j9H99j+78f1yesWdji
e2vxiBT0FzkJ67+rT36r9v//Qls8MRvr74n6Go/qx8b3ZyKPGee3iX8F4Zto
eQVYF7UnDl+G33cOQz2whv8MfrXn87ndGjTyodZfzXf0X9HfgiMfQhe9F+uv
Gb/Xo+d3c0cI/G/X/VD5/vFhsZ+Kjxb/C4Iae3cx/Gk/R59a37Ho9+woPDdb
W37+wOdnK7z3WTwHrPN10H5qLdZ948tZVagHvxn1UM/fWuZHi9BPkN9GtO98
8Qj6GRxT5ZF/n56F93/aoRHzWqM3i56Ph26udz6uy2Avj/YPemwc90er7u1r
S/3RIuYul/olPDJCYrG+mnFz4FPj6UD6vttPEO2Xw/+h3P+/UnsI7cdWXMRz
fvdOqBvmskbs152QYOi1+j4P9v/A0SvQ3n75EN/3fXEVcVlX6PRxm64+1P0f
OHEN2tdl4XnGmR6D/YtTs6JV7/+sGclH0U9fvnfQJw1v1fsHZ0rIx7h+tb9F
3TKmeLfu/UXaRLx3dNXuR13VggYVefT8NP6I/G4q/EesALuM6+MVvMfXAiY3
wL8ZH9Dv73VA1BvvKP1jtvdH3dD7L4JO+5sf68VbN1hHXi1nPBygrB/miVvH
oLt8DVHPv0+/xRvQmed+wXGHjGlQ3sez5+L5wuk8kr8THK4shS43jHWn0I3+
rKGaUm/cewP+0PcHYp9gRUTiejinx/J9Q4QX889dp57/wC54/2m7Q3LQ7n5K
MeLFUO6nVgdiXCvrttL/2rX2+J3EbJ6+CTH6deYxIxT7KN3LxfF7xin3P07H
2+uhG1JtYbzIYkTDeyV9fyoP36tTuUD5O4/pFboBx5P809BPr89RP+2peek4
vjWtEP21D85Wjl+ts976jl4Nfcck9GMfy2A/8/tg32T9ufc65fW3XaiT5p0d
a9HPL32Y/+YujAc28LpkHktVfv89sjMxTv1d7rNuX+O8t/njOjrdXuL7o/Lg
rSq9PqAv8jJPvrcT59f9DjpjQjLjC5v4vZaWKfO3K+ORl20Pgd5cdZ3t2ofj
e9HHFPP4nGdX/S/WYW1tXQPvUwP+tLwTGpHvgmJEy/G9jvOvbb+hvH+LYqHT
QlMQjZqD9Pm7iTzuDIVeG3CyUakv/YLje/+A7Ye6G7ju3mMdejOJeT3X87ry
+qecZvtMf7Rzovowj139mMceh77fnaAc39Uli/68VML6NZF5OBtq8NkoeJvj
DwhV5383jfqpuRyn+SDncyaH/V0/zvo5JEFd/2auZ7voTERn0yec9/1d7M/9
e+jMbQOVeufjhZz/qtXMN8CkPjOKx3M15n9umFof8ybzbRfDedxinTXX9uXx
/cHM71SUuv7GBlEXFsDzg6tZt7PLuG+b58t5ZfVU65MaWKcDnuG4Rd9n3m4f
jhv4Y+perVbWXzv/ezzf381+3ujA6zDTj3k13Od69K6PcnwncQvOO2EVzHfM
ZfYzqYrzuHiXcWWuuv53LeHxqlTul8ty2M9ZtteCv9bNczz6/fmR088fvtL7
daVPf9oEn+t+3W4iTk5G1DI+uqmc/1Bv+jK/GdEqKUQ0tCZE86kl0Bnly5V6
50o5fGVrAdR3aMc4KYP9RAUgH3PQi+rxy6jXP+nPeQR14Dye/Prz+AjWr6w7
yvrhzN7N8TuXsJ9E1in9RiXr36jPELX5Y9X6+s3Ud9qB6Lx1nLrUIn6u9eN8
lmxX6rWVG1hvo6jXmrbw81+Yl15Qw+sYc09Zf4wLH+K4MTmVuksp1L2exvE/
LeR5o0xdvyKSmH/HdI7fifN1VuSzvz9lYnyneaQyfzMjnvruydSvzuTn17IY
n73Mcf0XKse3s6Yx35m/YhwewX5eYjSW8ryTOE2p16N+xONPNHD92FLEuO8A
6+4Vf55P6KvUOxVu1selS9h+2R9Y/6MjWXdvb2XdLDyrrt953sxz46fUvcN1
0LWU0V5/muvRZB/1+iMIgtAa/gnIc87d
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 62->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmQ1QFtcVhj8VayQIJJ0yRDK6YrW2lqhBBaPGJcGfccZkqi3Un2nWTEI0
GtGkNcWpzRoEWk2JJJggiK4IKioCikI6BlZBo5EYxB9qauJCRAXlRwx1EmJS
eV87087cEPmYtJl6nhnmzO7e99y7l33PvbvfgKejpz/bw+Vyabf+fG/93eMS
BEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQBEEQ
BEEQBEEQBEEQBOFuJTGzckIX5MaSxz/qil4QhO8OLSPmTLs/tRcbzip9Wr77
Qkf+1cM8i0tuRSOjprSkC+PQM71elTohCP8lior+2O5Xe1HBRfhu4tlGlf/0
XY+c6dCXM8qu4nrotOZO+ffIh5Ht7e0ZWRmoH+ey693xvxbdrRR5olJru1I/
rJi0E1J/hLsN474z3J97z21SPf/awtdrVOft2T1L4LsDI1Ef7KSga+74x+p+
+RD0YUs6Vz/+NY6hUyradfqmV5Tj/zaMRY/mteucK/ldqh+C8H3E3uJNX51L
OQmfzPB889+fc+fYfVh3zWwvpf/0szdLO/KF89eoYuhv1rrlXyc0ObldZwWs
dktvnJ+M+uFq3VTdFf/avX/liP+F/zt6tVxuf66NqkV4vs3Ex6pQB64Mw77d
iEtswPv/pMGd8p82stSCb9umoK5YWQPdW/+XrdyN9Tckxi3/a9vtOOiLk4rF
v4Lwn9j9/M/D76On1sHvqfvwvm9fiKHvty+F75yHGu/If3ruTzPQ/onD6+D7
CKsQ+W/2dW//fiKB+t2Pdmn9NqMaEsX/gvANaMvhT2sI13mt9QKi+WUL1m0j
fFOH/jVCPkUdMcM8+D2g4Gus+/aKaHw3NIvT7mj9d/LS30L7BYHrMY7nDh7F
uGr8Gtxa/3P3fIjxz/35253RO5UVG9C+fNxh1LPkxny39i9+v3wH/b8w+XCX
3j+qxm2W+iV8VxiLtuJ3PDN6Jdf7pTvhV31CMP3/9Jjryvf/ktgP4NPkeLxH
2HuS8J3NaB2KaAdls47cnNKh/53IN7LRPv1H+F5nxBzn7wZe2cxz9cd3VD/0
2OptaPdxFfYxeoKOumGF3d+5/UfFkyfoew06e5DVKb3+yCf4vVIb/XvUP+e5
6536/qitvpQLXfkI7nv6PnzMHf+b49/q8PuMcHdjL9iL3/WduCH4vq19fLwF
z3vDNPjd2vkVjp3ffd2ieo6sp3phvdf9p8Of9uUc7iOWb6Fv8nfx/Pkmpd48
nM/vjpUvnEKdCFwHnRE8G34xQ66xfgxdrPS/PuVhPN/G8IQy9FN1O5/PVuj0
+B7Io/c+pdQb/htt3PfpGwdwn96LOR8+HtA5z/K7pZW9VKm3w2ZifXel5xah
3frg95GvXzjr6Lzbdef5eUq9dmkIvm/oX63IwfW4ooPIM7YPfG/2eoX3MbVQ
+fun8/r+LJzvtRLfW7SNTyLqa3bgfoxfD0BdN14eU6asA0mhW6B7r89GzN+5
/tj3mH6nkdeaVol9i5Wcvlc5/x470J89dOw61sl92Lfp81NS0H5CT+yb9Lph
mcr+K3LQztnqj/dGe2ka8pjGfkTdu3A7dLFvpCv7H9gf82alx6Mfred43Lfr
pTm8fyeyAMc+lTlK/fR49tv/EO7PGVzP5+C3Gv8PfoPw/uk6Fqncf1n2Xzje
lpN7uX62cd4yIziv767Yh/PLF6R+L+vwjqvwpdb8Itf3xMjPEC9PasU8pG9G
1Kte+4dy/o4EQmcs24OoDXyfdSOujtHnU+Y7E96qfP4/b0T/+rAfoL158YeI
ep8liHbrKeb1qPtM6b+SbrweUY885n5f6pexbrneHcXjUW3q/cuDZ9HOeOIo
5yG5kLoy5nNON98+DvyG/tnenrGH7TafZL6xtYhWSgvz/G2EUm+u2cbrrTs5
D6EJzBe1hsfzq5kv9Yq6fq5/k+c3vM32X6xie6/VPJ+3mzFyvPL+zXERvO7d
l+N4LYR5vII4jp+sRd3S6sKV/Rve3Tnvj/uyfcAY5lk7gudn96CuuU1d/6qb
uL+M8mC/w/sxT2kA9R/dw+MvmtT1t7qe46v8BdeZnKfYrtJg3kmzEK3JLUq9
k/gSdcZwtr/hz3ZFfan7GfetZtNK9fivNlJ/P9cH+94URCcglfkyLepjPZXz
97/GbuoGX2orAulzczF8bvh+gmgtWHgDMSThhnL+FnpBZ0+sRTT+UMHo05P1
Yvm90JmZwUq9kVUPX9gTukNnevqz7kQ247xTv4bjSy1T15/uTWin+0ylLrwG
x+bUBziOAubRYncp64++wcZ1K/496o4eZfvfXGQMDmX/12OVenNmAftvK6V+
5mmO27ecx0cSeBwRoNRb1/LZz+AtjLXFzPenjTwekMT5PXBIWT+s8g3MPzz3
ti6N9zNvK+d1VDbjrPNKvTF/O6/7lVA38gNEYy3vx5VXzHnMiFOO3/DNYv+r
C9n/tuPMM4nRNWgz5+HSJvX431nF/kb9mePIW8/jckbXXObXPVepxz9xLvPX
PMP5OjiH/cfM4vlxcRzfknnq+hvQj9fndKPu74xOWg+OZ9gA5q0bqNRrjwVR
92VvXm/xoi66D/svD+H5oNFKvSAIQqf4J6ajvYQ=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 63->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmX1QV2UWx3+2BG4F7q7TUjrptWi11SxfVqEtuJNvOa66aZhTjXsDDV9W
JRmjYS0vIiq+pCyoLNvKBUXKV0SWMnO5KhoSpAapRMYtSAMBIRCxkorv1z+q
eWLkx+5sU+czw5x57u/5nnPu5XfO8zz31ydk/qQZv3C5XNo3f7/65q+rSxAE
QRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAEQRAE
QRAEQRAEQRCEnyl2QXFRUGccFOhnOqUXBOF/hjbX+0RbfVoZ846r6lTXtn8i
9SsIP02s87ms75bgC6o6t167eqq9+jdXlEKn542p6kyfcDIvVbqjt3vfbLfp
nOnDznYmvjV42VvS54SfG3psxQdt33stPueisv6brY/brf+C+8raPrerg+vd
qR/nWCPOF7aPr1t6/XR37F+MoS11bsVPSsmEvuca2ecIPznMmuxa1Hfur5Xn
ePvE7Vi3dd9oZf3pI4fnt1cXRoPHIXye59PgTv2YtSP+1aYzP+vrXv3X9UZ+
+qL3OlW/zvE0t/YfgvBjxvIcwX35o7Hvow/cVlSC8dDx1Tj3l3qhP5gjN3ao
/swlcQnQf3IE/sw//s2t+rft5mys/93edKv+rYBzG3BfNb+X/bsgfA9nsP9H
qI8PuqIPOOGZWCftyftQb/bZ4bDmMo/P2qsf6/GAFzC/rmkt/IVErcN8v1N5
uO6/za36NeLPZGH9/seOis7UrxlwJFHqXxC+i9NnMNZHO+Zpnu/n56BOjcSt
rHuPB7Bu25EFyvo1fPwszL88C33E2XvWwfhKFtZ9LfRL7h/q9eta/+0RXYNz
vzXWnsjC+V1/eahb/cNs8DuJPAauS+uI3gq7ZRPuu+xgAfSzfhvv1v7jQ9/D
8NNr4tFOnT+SWzdL/xL+2zhHryShXkt8z+P7dTSY9X5gGNZ7o2ExxvqxYY3K
8/+KlXhvoKVtwv7BrjzN92w5a2Cdh8vQV+zA/e3Wv2H/dTs+r6ni+7rSJ2vg
N6IG/cOJCbqu/mGVeG9Dv0lfdQ55p6+HH2Ozf4fOH5r3n5GHa0ok90FxH15f
/3loRjjml0RHw66PQv5Ww8wO9S8t4VA68r5tOt476AfnFrpV/00JR6RvCD+E
7pWAc7+zZjrq3wyYgbo3F52HtUfegLrXl1ar9/9PVOH9v776AurLeu4eWGdm
PvuGX1+OH9qi7B9WcTD3CaWP43dEqzgFdWJV9aBdmAirlRcq69eIKcf66hx7
EeurUVb7HmzmU6y3ljHURw1R6q0en+L3QWvRebynNLsvxPNwJU9G37KyZ0Dv
dIlSx//dl/twfcyIffydYBXq1Bn7LPMPudY3ulWq9T7jeL4pe3kX5l8ZiPOS
3rob+ylz83bmkV5Vo8y/Zir6hD4qNAX/r2m9sB9z1TkHoa8OZv8If17ZB8y4
wC3wf8kzGXk/u2QjdI98jP2GvetFvjcZXfRvpT5vL3TW+y7sj+zEfrR3X43D
/HczdsNvj9BU5fcnth/Oi2ZidBL3WwNiMc6YCqt3mc8+OHvleuXzC3slnevL
3q2YZ0W+gXgT9sCa29/cAd2jEVuV+XeZiHOhdnbZq4i3K4E602s//N2fBL1x
cuoGld7eH7Ma+gP98DuRXR6yCn4ePIzr1vEWXr/p9Zd+jH1YO+GNutTy+8Oa
/rFNyLv6Iqx2U/ElPM/l25qV9/9WPfcJK5PZJ7pOgHVGhsNaiYvp5+byJuXz
L2thX1nazH7zpAf9DB1OP4ljmNfxAUq93dOL+e8sY9/yrIDV+9Kf02Mj/YcF
KvuPGVfKzw+9TZ2zDdbqto/5BJ3meOA6pV67epBxEwo5r3kL/d2bB6ulfcS6
v9NW6u1kxjEGpTBe5h7qQrKYT+/LHGcUK/uvNS+O+rj5tF/M4n2vncO8UunP
9dggdf/9y2OMs5p+nJ7ZvI+Iw9RHbqLf+inq+JOmU//VTuY5ZTfn+bzKcc/X
6Hf8TKXe/HwQr6+YxjhrIhi/cTb9jp1L/bxAdfxf1uL5mpn3MN6QIq43W7j+
uCqHU9e/Wdl/9cJ66jdU0OYcgNVS/8P5jXzvpX/uqYxvL6femtOFn49zuO71
pz89/27eV5B3u+/P/l9Yqd1Q33rkrbDG+mDUuf3GSVjjqQmX8TzGlV9WPv8b
WlGXjuc56O3sJFgzrIr+WpfDj3bvJLV+9xfQ6+EN7Du9R9HPrbUY24sXYqwl
Ryv7j76jkvqiPryPvxexTzxzO3WnGjE27zxzSfn/G5/P/J9JZZ/qXs+4496l
9RjA+5lSrNRrdx3hvNo8xk0u5DivlLaC92G+8wel3vF/nbpP/wlrhGbxfi4s
ZV4XczmOz1X2P6vXZvqfncJ5S1ZwXuVL1Lcu5HMtVPdfxyuRuowNnJd1bbyA
fdt++gz9lMxW5u/KieLndczXiGd8522Ota8q2ccXPKeM77oxlPH2GIz/Shj1
TdM4XjmXfhpClXptwdhr8//EfJ8P4ryYAOYzbQ79jJqsvv+dfak71Yv25F20
B/w4f/Fo+pk4RJ1/5P2Mf4cv50X8hvPSutNPQyCvp+tqvSAIQkf4GlozzDs=

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 64->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmgtQFlUYhn8kTSxJjbyMQ24XDSNvkSOVDKtURGpppjlothFOFwvFEdMy
2FLTMs1UShN11QijCK+hRbDGTUwIS9QycEWllEJE8DLRULyvzdTMieSnZhr7
nhnmm7PnvOfG/37n7A/XRUx6cIKny+XSfvtp99tPa5cgCIIgCIIgCIIgCIIg
CIIgCIIgCIIgCIIgCIIgCIIgCIIgCIIgCIIgCIIgCIIgCIIgCIIgCIIgCP9X
IvZkBzdDrn9UWNwcvSAI/x6GNjGnwZ9acUmWyqf6C4O/F/8KwqWJ/npCeYO/
9e0rTqh8buaf3d+o/6e8fhz67PvKm5UnckIcd/Ta5Iq0Bp3z5tSi5oxvpyze
InlO+L/hlK0pafjc2xmuKtXn37g1oqwxXzg9V0NvGdUn3fGPk9KlsEFnth1f
6Y7eLG4HveXTwq3xXR8fTcH4FZcfE/8LlxpmvlYBfyfckac83+NH4/y24gKV
/ncW+hY06v/yh228P6zresot/x6bOhvz6xbjln/1VZm7cP4vi1LeXy4Wa+WQ
4+J/4VLD7tqK/vaZ8DXu6ef8Ec3dH+K93vGu/xH1JduV/v8rzFa7ZkA3aTne
D6x1lW75347KwL3b9Pesdsv/B195F+uYNbtZ939BuBSxj207Al/4xv8A/9eE
4p5rBiXC72ZyFKIV1Lj/9HnT3mqoN9p3fBvtPrsqEue2T34O/evtlv/1ljEb
0W/nnKPN8a9zV+UK8b8g/BnT22sdzscTeXi/tp8Kok/n1NP/pWko25uXKf1r
P5uH89UoGHUY7WZtKUUesTfwHrFoLPo1e7dw7/yP7LgH+q12k+4fv2Osj/kS
8wtcsrEper3ztDVYT+07jb7f/B2mdzr+buKMba18v7pYjB4L1kv+Ev5ptKxQ
+MJOfgTv19oQHT7VZo7CeW+96YHoLE9Wnv923Ilc+D04Ee8R5upX4Xd9/nvo
zyo24Vu9rrZx/+dkJ2Ocsp64p5u/ZPF7iXfXoD9HG3BR93/jmhHwifNkMe4x
ett2P2E9P8c2KX9oS37G94ZalwCuo83ai9I7L8eO+WM7M/YZzN/yWNak8a3c
URbmP+Ub3HvM8Alu5SEtz8mRvCH8FdrclO9wPiaX4/sxc+zCap7j6+g3p+9p
+KlDndJ/5jv1OPfN/O68J1Ql4XNuD+qBspkaiqhf0fm08vvDipN74bPnrzsA
nzzpyfZbpzNvrP2AvqmKUucf3zD8/5Hx0OSd0M94fh/mO596V5oB/zp9rlfm
H31arx0Ypzge57S94si3aH/qU37fuCgO/Rg7xyj1VvX4T1Af5s+YXb0b/XVP
hs45d4b74Retvj9Vxm/C8zcWbkD7e1fy/6mcx/B3FSv8Wup+SVT+/cNsWZ+I
5+FZuK+4Bh5ei33PjMV6rFLmD9fB87nK/Q8tfQ/zrR8Mnd2yH++Dh3p+CP34
knz8fqybtyn3L20kxtUye6yCzmcxy2XJKOsl2Zsxn6zDicr1n9yE90brfH+M
b7Tqthz6iY8zvjSP89j0yUplHotMSsI4Nw7A+WF6fZGOdRR5fIb2UcYGnktL
1fen0fctxvMtHtgHZ1wG1mk9fud2xEPR7+P5+r1Llfuf12k+nk96IxXtPvd4
DbqkwXhuXJmL8bUWZQv+i3nY6RYEX5ozWiJqt+g1mHfbQ4hWxpRa7O/+988o
5391e+p94hHtF7ci6kGdmDc8K9hvv7Qa5efvmVPMNweYH6wsX+oSHN4/0ryY
N+pHqfWty9FO6/8j28+uYp7Yzee6VoxojJijzD/G7k85fk4B2/eLRzSTSxjb
DOK8nhukHF9/rJC6vukc3zOL96X0TOrHcB+04SPV+e/AWuoHLuI6HplO/egE
Pr8jgP0EhCj1VhTba0kL2H7HXLb3msx+OrBfV+CjSr1rZgTXP/RZ7tNels2U
cVzPuFv4/GC0Mv86Vgj1Nwxju+GhnEfwUPYTFM74/Qh1/tb9Lozfk/ONuZz6
sMsu7H9/1sf0Ueq1Vf5st6uO501cAaIVwqhv9OW8rhyk1k9tx/2LrWH71HxE
o8uFaLLeOeqnXr8f+7e/PMM8/Q3HNXYU8Tzs3YH1Wbe59f31v415Uwj8bbb1
RnRFl56h/75ifLrXWaynt35OOf8Zx+EL52wS9E7dnYhaQBTLB8LZz3qvs0r9
kTrmGVcV846nDp1d0Jl5Z2Ivlkcer1XqK/dBpw07zX4OM2/ZUbUs7/Gl7kiq
Um898DnbFSYwbi5ENP0zOZ8dXtyffU8r9WZlLtrpfhmcR84W7kct9Vr2Eyx/
1Umpd0q2sb7DBo6Xnco8M2sZn/eZzH6idyrzj1XEdvr5C/HqBVz/qvlcj+dc
ridws1ofHU9d0dscb9oalh32Yx2L4fhD7lHOX/eJZb0RR92QORwnbB7HTe/K
+ukzlePbt0ey3cCHuf6vWXb2RFDX5in2e3ekOv9+dD/n+eJQxuHD2P5EGMtx
E9hf/7Fqfae+3K8e17J98A1sV9Odzyvv5jz2Byv1Rn0g6z/2Z/2AAI6XeBvL
HR9kfd29Sr0gCEKT+BVuurOk
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 65->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmgtQT1kcx/+yHssWmdYjhtsuUww2tYNdr2vllV1kN/KcK4+8Si221cje
YkMhWRVJbl4RS2KFla5HoaWSR/K83oxHEmHYsfp97cw+jkb/dmeN/X1mmt/c
e8/3nN+53e/vnnvKznNiv1EVTSaT9OKn5oufqiaGYRiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYf6n6Gd3
7uxUDr2SsfFYefQMw/x3SEUXr7N/GebtRG879EaJv5Ux9ndEPlenZZ8qzf/q
nmCqD/IW70vlqRNqYmKp47wSm4ubS3S6ff4WrlMMUzZkry4XSnyjXptTIPKP
sndgqb5W19kb5L861e6Z4z99TYWsEp20bdlts/wbZPEL6cZa3jVHL9u4rS/R
GbfaXeb6wbxtqP7L6P0uDwjSRc+37B91k97/iceF/pdGNzhami+0wFb7yb8f
h5rlf5N1rdVUP75YbJZeGemO+nEl0iz//45qNahceoZ5E5Gr3IX/67gdphgc
Q/txUmjxFXrvj25O636l6FaZ/GfEFnqXtNdu2OdSrNTsvln+SVqYQnn0X1to
jl7zdUqm/NPjT7J/GeYvPHt6lfwxJ5Wi6tGcopRsS37TD9SjqOoTXst/avGl
SGq3IcIx7UXQNo3LoH5mF5jn380+iyif3kFXy+Nf+ZhpDfufYf6MOm8m7Ysp
ak343b4dRalrNN73SbXh20quwve36p+nkT7AlvYBtMwmp2kdEZuDdUSGBfVj
rN7wWv5Xihu4/7Gd0d+Svt+lRE/z6kd4L/o+kYYOSy2LXv+177iS+qXsDKTv
IjV7YJxZ40/euI3uS+sdZRr/b1T1j+f6xfzTGOPTdpN/Z9THez4zCz67lUB+
l8ZaUdQ7Lxf6X89Io+97vUIl7BP0WU37BPJWb+wX9HlE/te9upe6/pctC9dh
nOHkd937OfUnVUun7w81NuS1vh+kFcoKymNBLu3Xac06Yd/QbZJw/+JVqK5W
2TTuvR6Uv1zc+bW+f7SQsVpJ3TANqjyL9FUdSKf6WJRt/2KhHEV1s2ONa6QP
u5Jt1v7pkb37uW4wr0Ld5kD787LnHdrfUtZ2JJ/JXeB/7XHXInp+Tt0Qv//r
2l8kv9brSXVDGVxAz7m0NP3l+/ohzju0LxLq35l7gp7vrS3yKb67FfqsRPhu
uwl1yWOOcHxtbj493/KY3EOU79SMPNKfRD3TO05G/cl7KvSf1GL6HuRbGf2E
PTlDxzcawLe7ulM/xsk44fpDzb9N//8k2Uz8mdqFNaT9RkOdg3WP8wCKSnyE
UC8Fn6b9CS1gTxLladuBvpeMy62pfkkBM0inVXEV5q/a3qb9UcUlitYHxlQ7
qn+6qw3Nx2h8iPZx9Kv1DwrvX3gq6U3B9qTXd4dhnRHSCX/3OF8pE/0XpAjr
f+b7y0n3mcVSzCOH1kmy87wYOj/CFn93He64Sph/t5xwus8zu8XS9XONQ0jv
nUJRqm65lvqxiooWjn/kFK0/9W9kGk+5GU3j6Q0zERc9Xknx1DPx+qmH5yTK
u2Ub+m5ValzYRPkk+tLvQ5OyYjE/j++E+T9xHEHtZtmsFl1Xpo1PoP7iA3zf
xDqsFHYnX0rh4ykqP37/gO5/11sU1cUBDymOzy8WPr8+xfBll+qk1/uvoSh3
nEzRyMOx3rLGQ+H8h55FnRllA13LGogffAX9/OfoPyjigVB//yDWKS55FI22
NaFvfBjrll1N6VitNkGo1xv+BL17KtqvX4N8Dh1HHfzSRHptfW3x+PkpaJcQ
h/E7rES+vpiXvGwTYliOsP5JLj9g3NQYiurkBdArOJYr+qJfP3H9lHOwLlLq
LkMcvgT9KDhWHYMxvwOh4vob8zWu+7kjOk3BeNkTcR8y7uJ70G62uP6/1wPt
4lwQB3ZDPwbOy00/ge7TvuL1o/XL814jMO6RgdDLI3F+ZCDOP/US1/9rCu7f
kqmYb0gExt0XifbnFuO+XPIX6qWWbtCdxn3S5wZhvNAZ0DmtwvX0sUK9PBfz
k6w+QvsNjaCPQ9SsukKf/bl5+9//MpqzE/lSWW9HUU5JIJ/rx93h9y3Oj2ge
1088Eub//BH5QhmzhPR6Un/0N30SRaPXfYrSYEehXvZ4QHppcRVqpyU3Qr2J
PY3zMyujn4AAYf2QPzxM7QzTGfjzXBHysW0C3TBrzMvPQaiX3HdC77ccOscd
OK6aRVG71wL5nPEQ6xdtp3a6Rxp0D17q26ejjp5ZhbySrYV6JX8zru9LxPhV
tqK/zBgc126F+xB6VFh/jEnRGO92FOKASOQdG49+ziMvZe0BoV4pWoDzx8JR
73tCbwqaD13CMMxD7yCef6I/dM2mYNwTAdAHB+J8VAjiIH/x+J3GIM/cwWg3
1BPzTRqH+bh9i/PWvkK95tMX7S92QPtaiMqQzoi5Q6CzU8T1f3QbjG9tiX6m
1EO7LfUx76xmGN/HRVz/3WTomjggJjtC5+6M2K4fzq/qLdYzDMOUhd8AY9eT
xQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 66->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmXlQVVUcx58b7rgvuN5GDQXTMg2zMS5iiopghgpqdgdU1HGBwRWXbi9L
RcWNAhLlqgkpieKWuF5xl9xSQhb15gboiIogkY7m+339o5ozFNeaGvt9Zpjf
nPfO93cW3vf3zj3vFf9JA0dVsFgs0rO/2s/+qlgYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmFMobtH
/uD6b0+CYRhTKNsD89i/DPNyoiVvumnzt97ITuhz9UhoRmn+l9NcbtjeVwqD
sl+kTsjDO5g6JyjzE6fbdNJxJeHAC4yvzR85kusc839D67rTIP9PWXNX+Pl3
rXO1NF9IZyvT+1J+H7H+T9CPp52x6Yy9ve+Y8v/ZZvtsOvmzAffM6GWfvFU2
nTZqYRb7n3nZUNLsc0v9XNfaS9/7eisnoX+1CZ4XStNr/Xankn+c3Ez5T9rb
8zvy78IMc/719qJzg9o9z1T90d68F0Hnl9Z38tn/zEtHhcG3yN+b5x2kGFbz
PH3e7268Rm3dC9+7ARvK5D/5qbWn7bxt9B+fRv5/0u2+Kf80mayTLmOpOX1S
MdYVUfHKi/hXO7wznP3PvGxIaeXo+V4J60HRiKtGUSqZSX7XAhtQlKvLpfpP
Ssv8lPr5KmN/20/tVPckvf76jAJT/snaHAz/5t54ofuDgtFb2L8M83sUJ/cU
8ldKKPlc39KEfK5dqo6YGI/vfTVG7N+s0SG273m1RkN6ztfPd/qR6kdIYzpH
aFoCnZuVyD2mvr9lX/vjVJeatjdVP+Rc/QTNy3nzQVP3B7Pskk3dW0QfmUjz
jnHcROP3ur/LTB51ih5A+xiQrXH9Yv5upGt25C95mnsB/FsePtt4iqJyywvt
8BVC/0ntPqf6oVWvnYdzgkL1wrCk0PO2ejOB2pK3b6n+VYKH0+fb2FXzGM1j
z2K6l1A6VKb6IWel/6X6ocmLl9N4Q51w71ip5DblS7Yv0/OLdDmV7i2MaQE4
B91cUya9NsjbaquLyvVXUVf98st0/yB5DAvG7xan8PuJa9ZZU/cXA9seJn3/
Bku5fjB/RPq+4Cf6fKePIn+plyLh+7uOD6id4ktRdhr/QHj/Fx5Cz9Va3b7Q
//IeRcm6DH49lIN24hShXvW0p/OCLPtkUp7XOlN/ZdlD1I1OTamtl4sV1g/Z
w/kI+bSdA/lVDT+UTroiB4w/sD3mNTtEWD+Uael0LtCO9CSfyLur43fKRl1Q
x6J9MJ+mI8X60X57KH9iIp0TjPuVTlM8thfzvyeRzghsJtTrK+VtNG5FjyTq
l10f61lkpfsXuSAHeTz7COuP5rg/jsY/l7CO+s+YsZ5iQz/Ko+d2p/ohL0w6
KfT/QJdv6PXTQZTHkvyI2nrhOJqPMtSe9lXu1V18DpqrraX9K/6Sxresuk56
OXREPMU6k3dQfDIgXrj+MHfcr94YQHrd7mYUrada1Whad0QSzk+HkmKE/3+p
Go2vpDuupn6jttF9sZqRj3vjyxdpP7QmvuuE+k0J/qTzP0z91NO96Zwmp1+g
KLXuspHytxkxTfj9N2bFItJdXUvPl2r9i9RfG+uQQP0X1NhK8cq5/2T9Nax9
yZfSjVqFtA9tUxCHfFhE666XTlHyO/FQOP+4J/Cl5IM6MbEDRW1dPPLuz4Dv
k1sUCfc/8xDqzbVLOH9YP0B/u4kUlbjJFPXFZwuF/os5QTr18TbELucRxzXD
fC66UjRS1ojrT+oG9Pf8mqI2fRXmkZpHUc4Zgnl4HhTXP2s89ZPeWAndgn3Y
j8tLoC9CfiPDRai3HF0NfYMozMMX89BPpCFPxdoY/5GzUK+vno1x3p+JcSpM
hV6ZjH391Io4IUCol9yHof+gjlh/y2bIc6Uh5mORkL9XkLD+KmOckH+ZA95v
WQ953kbUWyKvavQQ6o2cdhjvDPobTpWhe1oTcUdnvP7UTXz+7NYB+5fTHOOW
1IWucQu0P3kX7UXuQr3WvD7ml+2Cfj7FVGfVpBKc++bi/Cv97CjWd2qD9Q/p
iufli+Upyv3KoR3sjXb+O+buv/5h9Met4EsvR/g9eyb5XCtqT9Ho6lRMrzc/
UCz8/+U+hS8PBpJe/jaAomFdgfaiIOQ/GijW+98vRL0soijfdkX/zpiPMakz
ostlYf3QVdQrJfYq8ixBPiXUHv3DHBDd/IR6NSiJ+kvbY7COa8WISVdQBx++
hXV4aGL93ESMN28f8hSkUjRySvB6Ltahd5so1GstN2C8FI2iOms92o2+wn64
DUa+sIfC+mfxikC/SoiKfzTyxO7AfPbser4fV4V6qWo4Xs9djlglEjrntRh3
UjzyewaJ63ebj9E/YD7Gbb8Q401FPi00FnFzmLh+15yD/plW5PHAfOQCrEdd
gH3Rh4n1ljOzoUuMQqyP/6dcZyfGjUKUS1YI9frWSXi/4xSsNxbj6pnYR+Xe
F3g/ZY5QL3/kj3muD8b7fZegnYD1KznP9c4h4vkzDMOUhV8BUFCftQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 67->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 68->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmQlwTlcUxz9iiyKMfWk91dqmqCVfzCQmb0QqGYIGQ0p4EbGWkNKoIM+S
6IQGbRGxPbWExBaJfXsI0QolURGiebEFESKSRqmqnH90WnOr8ozWmPObyZy5
33f/5977vnf+776bRoP9Pf1sLBaL9OSv6pO/ChaGYRiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRhTKMt2
JDn/35NgGMYU+sTl17h+GebNRG2Xd6movjW3zpmiOtcz1l94Xv3Ld5yNou/l
Pe1e6jkvP8zcwT7DMP8t2lY31P9upzui+pPuDBH6wp/UcLhc9L1+p3OumfqV
ghoeLdIZZ/QsU/qpbWcU6ZR1Pqb0T1H7Z0Sy/zBvGnL2givPu69VPTyb6j/z
K2H9WyY8PP88veZlPU3P/0ZpYv2/MfnuYfIP2xum/EONzEqm8R9cMqXXnY3o
Ip36u5MpvZb6aQTN/4TtAvYP5rWj4Nj1v96XsnbrGD1vI+PT6L7tt+M21U+A
u6n7X563lPxBdgw1Vz9+1kR6fr9b7a6p+iloTP5jWZ113VT9z5ywhuq/jFfy
y9SvPtu6leufed0waqffoHp3KbiCffoUOs9TEltTvUrVEqj+NZfAF6o//Vie
J+nnb1hFuhFlT1G7dMU8U/XnuGQx1d/j6Jc7Z9wfc5Drj2GeIb3rSapTz21U
74bDMorq+MkU5cVzEbsPEtavmh0xG/trDzon0AtdUkjv9ICeu8Ywf/hIyNgX
8g9JezD2b/32HNUpb6tzpp7/qjKc3h+UNsMSTdV/RIBG6+kfG3agJLrmHb4p
6m/4td1aFNUuM78z5X8P40aybzGvCj2kxlmq3y5pVF/6olpU56rf1bvYt9eh
tlzPR/z8nuq7h+q8V82bFNscp/d8o311qnu93EnEpH/QPyVoaSj5hHMi5dPi
t+F5/9PVHJpPpSklqn+l1KOLlGdOLdrfyLE9Xuz9o0a5QBrPpRa9B+nedvDF
cTNK9P6ie7muIb8Y5EnXQ7ILKdH5hxKWRf6h1hyN85mtNidN+Z/DrQO0jrAB
gewjzLPIfdOx39/dFPV5zZ+isfc22mU871H9WJPuie4fI30l6izhAPxj5mCK
Rlpl+Mb9PGor2apQr+X+Rv5j2WeTTnqrchf56iGPYxDiro5C/5Ae7TpC+Q85
naB+HbqeozzDoFdLHUe86i30Dy2lzyHSJ/kk0Li9etJ6JGUQ9kHrWmBdcT3F
+ovWvfT95pG7qX/nBKpTaX8V6LrVxHr2jxHqpdZGHM17TMdY0sfOo/VY6vai
upf04nMT+9VCvTLpwyjStZy1msbJPLiW2nPW4v8mhzZepbZblNA/ZI/wdfQ7
JQ1Ann047zSOONF5hZJX4Tit48TFXUL/2BRA+xqpclU6J9HL19hA+S6pMaT/
yHUbfe74fpTw/pEC51H/MuuxPxpkWUjtmLcoWoIzKZ/WuqHw/y9GC4fl9HnF
yt9SvOhL11Oato6i8t492r9pE1M14fybJ/mSTw+Jxvh17OPpOoxIpCh3OEPz
1mMSBgr1LfcFU/9mORspln2EeHn4JorDtm+h8VMvhL6O/quHeKMu84Py6XrV
vU9RduhYQPPfsRDRI+YX4f1zvhrpdaUJRePnXLQDjiB2skc+eXGBsH52nobf
VLeFz2zvh/lkLaGo5sQh75ycfOHv33sL6ZW+OylqMxsgz74uFLVHezGPFl8L
9Zq0inRS9mbMY2gy2kfsSCctCkKe9bWFejU5Cvul/nPhd2s2YB6/3kC+1vBP
bcBeof/pLb9E/9uh6B+7BD5X+nvMI+0x4uRpYv8cEozxY0LQL2Ue8jlryJd7
CtenUT+h3hI5CvNuPwbjRiOftmIWPu+VCr33RqH/yg1GY/xUb8Rp49G/7FT0
HxmOea2IEOo1qTf6Dy7Wn+2E/k09MH7o52hPHCvUG3VdsM6zEzDv+7gPpM92
IdbH72JkiPefemR59OuE+cpNiscf5Y7ouxR6WRLrT+fAl4Pbo98nXliPky/W
M70nRX2Iranzr1eOvRX17T6W6ltLjqOoT2xQSPOd1Qcxdn6h8PcPL0N6PSWc
omHXA3Ve2Y7ySJ6u8A23gUK9dimf6kqyVkce+6EUtWzks3zQCHkaVxPqlR/i
SK82LqSoO8G/9IzppJeiLBi/SqzQf6SIDfCn/Ufz4R83oV9ug/X8mE5R7h0t
1CvWldTfCDgEf4g/DB8d+TbGd52DPKmrhXpL7QUY79xSjJ+WgPX46Mj7wBF5
ztQU6y9Px/WLnwJ9mzDox0dhHoF5aCfcFPtX/Djo3H1wHeyKo9sozGtSPPT+
w8XjX/eEflknjHfaBeOk9EBsFVy8vi/E47/TDzp/V6x3ggfGC/FCzBgOfVqA
UK9fQz9t2seY982u6OfXDbp8rEOzGSfWN+uD74PqIxaWR7/elZAvtz2ur/1o
oV7JdcD8W9qif/wVPK+23YLfPmyKz090F+oZhmFKxB/OQahl
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 69->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmQ9QVlUaxj+lUqvV1KxVs72SmYasY2OO7ix5cWxNIYdQVp22uLpamIsl
ViBLeVPBEBEVaijTrqj1iab4J1PJ9URqiBj+XZcN7RqlAraKgKS4tX3Pw87U
zonVz9nVad7fDPPOud953nPOvfd97rmXruOejZwQ4PF4jO//bvv+r6VHEARB
EARBEARBEARBEARBEARBEARBEARBEARBEARBEARBEARBEARBEARBEARBEARB
EARBEARBEATBL1TpksKB13oSgiD4hRPX8SupX0H4eWJUXXBR33sP/V1X58bk
aWVN1b96+ovPfL+r/sHqanzC8hTniM8Iwv+ZruHlvrozWw87o6s/p6zEbaou
jaAw6NXvR5/1p37N57MLoPPm+LfPWHPpDYz/VM5xv/TnZs7z6ZyIr/O2+6Nv
xDqxPEP8S7jeUEN+Wd7UfelkxVXh/r+1s7Z+nbktPm9Kb5tn90N/l6v1j/86
v4pRO3w6q+FBP/3jtQPwr3Yf+zW+XfnAWuis4n/4Nf+U9g50zr1xV+MfgvC/
wK5re+qH97X76osfot6SN5fieGg07/sFt1xR/bll7SzU3fSKo8i3Mt2/+h22
qAh5er3jl96omXfYpzPazfCrft17vsyE/vzWo1f1/Haq18nzX7jecJfNRP2b
j3/I/XW/4yfwvM4JRL25D19A3RjzZlQ3df9a09Jt9Eu0k/CcW3nei/baIjz/
3ZELm9T/FGafreuxf492Tl5N/bjDS3dL/QnCj7F+EXEQdbF+DOpdPTCdz9lZ
yxGtqNNsx8af0+7/V5U6vnp323rwHmF3fHIf/OSki+jmLsO+23njN5dV/+7y
3GhfPmP2olTkS/0D3v+trHv98g91z8V8rKvDrh3Xsv7d9rWp4j/CdcfRnvh+
b49Yhfpyo55DdAKzEe3NCxAN+zFt/RstUt/H8UuJlYijFPzCKZ6DaH5VzfaG
IK3+36iJt+L7mPLu2YJ6L0niPiR/CPYf9ra3Lqv+jU874XudfaQc/3cwxs7C
/sYZ3uOy3h9sO22Qz3+s1zvsxPmY/i3HD5l0WXq1rtDC/ufoyHxfVGN+Bf8z
68u/vpL6t59YnYN5BG7Hd0sVuanIH/+w7gvYjPm/0zVa/Ef4T1T+jRWst4Wo
T/OmKESn5xREo0XXGsT9q2q0z/8BCfAPd/SjqE/1zQREK2Ep6zWsOfNN6qvV
2+PTjnDf0BLv12ZySaNfjKQfxbxN/7k4Vesf6sHTqFNnzk2fot+Oyr+h34RU
5in6YzX3Ixt/wj8ysb8wT72C/YGaGHsM7YggruPU89RtX6DV2/0S8L3E6dIW
+wx3ZWAJ+rVZy/enqDCuo9l8rV7NDdqIcQc/ie8DTnnoLpyP3aVfoh3K82mM
WarVuz1mvYt+E2NXYN5z6tG2+9ifcB8XSR8tXbdPO/6r41fiePsGvK+Zi7u8
h3Unxm6ArubwHkRv3lbt+avLWobx7AiOO+hmfC81XypYg3bG1E1YX0GeVzu+
CsD3FevYNMxftfn1Er5/BiOqVq3ymCdzsfb50y0A/xe2w/mdVb3e7QO0W8cj
qv0hmJcnov1y7fUry34G8x0Qgvm5nf8Mv1S1t+M55AxbgfNh9plua9cf3CsF
uhYv8/ptGo3zZh94E1EVPsXzGNdt3vXov+rUJNSlHVBRi3VWVSF64hfW4boM
jmVcmnhee/3692Zd3zGZPlG4hflyWyGPXdOZMT6rTnv95u6lzxTTZ9SxTOYr
70HdjQMQjXaTtHoVuI2+NXAPor31PuYJSUN0zb7Qu/VTarXXP2wNdXkbEa3k
IuZLp1/ZMe8hWg2Htf5lVmaz/+Q5iG72Yfpd7nf0q9b0TWeRXu8UJbPfX+Yj
qpNZzJNQy/MSPZTzqJ6v1Vu9XqGueRrzDOY8rHczqJ95jutrk6nVq23jON+W
v+M6KoZw/NaMTv8/Me7P0fqvWzSI40QFs1/yDRwv5mbmCWYe458xev8eTp2n
rh/HP9eTeU7zuF08gnmChmr1ZgN19pC+jFNacv0FtzPeHcZ83Ufr9Sm38PeM
cMbmu+n7d+/l86zP08xb1UO//z10G8//OC91+Z8w7jrI/XPix43t4Cb3v9cK
480o1lVdAerbLChBVGdC69Guj0R0Bx+o196/3g7URyUhmne4iG56d+QxbshE
tCpy9Prw71iXxp3Q2StmMNauZp6dj3M+3mCt3hy7BXq110Of6t8W0Ti7lPPq
3gZ6p6pe6x/mvvX0mSP7mGf1Rfrg4i7o7/Q+wzh/p96/KpfQn779iHlm/RXR
Cbqf67jfS/2GVVq9/VkW+lsj3maeKUWcR8h25kuP5HrWRWn1lmc2x+ucQn0H
5nF/26hXQdCpEx79+hfO4HpTkhgT0pinmcO8l8rYjgvX6t27JvP3nhO4juPP
Mk9H5rNePsE8Vdla/1WpseyXO479suMZP0jmOtJTqYvL0OqNpGc4/jcxXG/v
Fzj+nTb1G3hePEuStXonO4Lj1TzE8z5+KOfTicfthlHMMy1GP37hI+z/Qij7
jXqY/T4KZ97ujevqEq3VC4IgXBH/AtPRrlc=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 70->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 71->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmX1UT3ccx3/EdByMFctDuZmVhdjyMGEuObYdbArZxnQR5eGgPGxpq1tR
J6kmY6xwe0CFzEM0kuv5YZLHLRxcZAkh1TzlnOnz7h87X+HnD+Z8Xud0Pud7
7/f9+X6+93c/n+/3e7MfPcVjrIXJZJIe/zV8/GdpYhiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGMQvt
S4tdvV51EAzDmIURe/Ui5y/DvJnom8vPVua3Zlt0UJTnxtbYc9Xlv+E99w+u
Dwzz/0TqHF5Qmb/GxfBbojzW/1x6vrr81rq4kV72TBPqn4UeX7yV6k+RjfFS
dcTr8j6uQwzzJGqBS7V5pbu5Xq+8r7z//W1RPzWn4nJ1erlhvWOV96UjXwn1
z2Sgxc5KnX7NxSy9GldB3xcM5xFFL5P/it7zGtcP5k1DH9mp8In3em7TQGoX
lf1FeWeZepPyd1eHF8o/I2L4uh2PrZzRkfYHmtNps/JXW5R3iMZvcdIsvRTS
ms4nWs0C8+qH7doN9ByuxL1c/fAq2sH1g3ndUEZ50Xstxx6/QDYi7irlm5Mz
5Ys2dA3sJ1NLqj3nn0hNonzPPBJJ/T6ITiQ/UR1o/ZfH/FKt/mlI7kGbSd92
1g2z8ifEK7VSp/oH579M/umP7LI4f5k3Db1O9GnK73f6UJ6rl8Jh91S1x2Ld
lg8OuCP8/peTMZ/u+3nROV91V/ZT22EoffeTPBfSuV9e2+258l/Z/0ih/Xrn
22Mq64nyw1vbyO+K59P/F3nh9U20fq/pup/zl2GeRKtZn9Z9w8af8kvNd4e9
kEtW8V5JVroTKMx/U0f/AFr3rb+m7wRas87YZ0cnkDXqX0f9mNFGrK9C7ZIe
Rnna253WWd364RXS3dsJP7FHniv/1bv6aOqXeYPqmm51kPwYG+1vPlf+5x31
Jl2rm7Rfl3vuon2HEVT8Qt8v9dRW/vQct/SEvu6D4hfRy5ahMRRHg2z8PgeU
vWbVr46HfLjuMU9DD8D+Xu/rRvmpfhGJPHXIIqtHOJRSu79WKnqPlNC3KM8k
29OoG0oN+KnzEO0AF/hp6SnUq94xtC+XA9Nx/mg5swTf+06h7qw+gXbzG8L6
oS0dTN/11VMfHcU5wYf+X6nkboGuEeJQnO4J64fsY7mb+v1mtxf1wo6+hxqW
yahj+VtJpy2fINTrPcqz6X5UOfYp9z0pDunSWugbF8K+FyTUKzaTN6HOBK0n
XdcLNB8twZXqlml7Iurnbldx/Kf7rqLrfmdSKH7HjXTeMa3LoP/XSjPG03lO
G+B+Qvj88h9AP73PSpzbdqTR/L0V+u4hZ2cfpnZO12xh/I3b0jlPXxGfTP3n
jU6n9p50sqo2KpNs2tJVwt+/kSvVOfXeliQap9nJxeQnd8gSag/+fS3pIj0T
hM9/QQGNLx/ZR1axak7rhxamZmE924f5TNyYIpz//FI/vO9tVlDc7U/QeVPP
jd5C+vN+q8n/ON85Qr1nlxDq55adQf2cB8BqV8iaKtT1eB6fR76OdVivkUd5
qQxrWk7zsMouo/Z9i3LE7Qqb7/KP8P2r4UF6LX0ZWWlxb9LrM78hK9/OIKtN
1cuFv799IeW1MrEd6dWYdYhnwrukM+ZMJqvGTxLqlZLjqDfdqmySC+mN4L2I
x9EVel+/MqG+33bSybsOwdYzYXzrAFj9AVm55hChXqsFvR6ciHlYHCYrOQ3E
fDbXJZ1UeE9Y/+S5y6E7Fg+dz3rMY3g96q9P9UMcfVOFeulGFMbPC0b8ZZPJ
aiuxX1PLbsH/WTtx/W7yI/QNfNHvp9Gos84TEE9aDvzcShbWX2X5CIw7bhDu
S13QjusFfcos6GuFCfXS4v7of7sH4nbEOqT36kPW2D4Y7dRx4vEfDsH10hbw
Y9MQfpbZYNysrrge5S/U6762iHO8qer+Qaxby45h/ehnBz/u3cTrz3BXjNfo
bcTZpADrRa1i1OtVjlXr6Mhq97+vCn2sL/J7azHlt5L2M1k5UblL171+hVWu
3hXFbwQ5kF6qyET92LEe/oY2hj+PENSNwgNCvclUjjyPdCOd3CkA/qaXkDXq
+EPvM02oVy0PIC8/7kj9tYUdYGeHk1UG9Sa9Ubu+sH7p3behTu27Cz/ta2P8
UUMQT89rqDuDrgrrj9ZnNfL708vwk1eC+mnljPEtjsFfaK64/nVOgL57MupU
m79hs84hnqQU6DNjxPXvzDz0D1wAP2eW4HlG7EY8w5vg98i0EsefNgf9i8Ng
naIQf8QijJ+DdUBJni3UmwK+wzhh01H3i4Og85wNPxswL8k6SVg/pdxpuH5r
KsZfEYj+d0Ixr9gYrB+lcUK9bD8F4384C+MnYz5q0wj0v4R5KK3nC/VqA2+M
224Y+o1D2/TZJPgbhXVDXq2K9ROxzqnpHoizlQJrPwH9yzAv7dtpQj3DMMwL
8S/rwYz8
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 72->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 73->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
eVa/SF0V/27pBZNY35HX1PMLgiBcD78AAEWYig==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 74->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmXtQFmUUxr90SqgRr6U2mVveL5maWVaja6bjBS+kjTkKbo4aXkgFCQwv
m6h4Fy+IoekSGl5QI1JKKzcNFSuTNE0lWwwREhLhU1FL0+fxn5pXwq+cHDu/
GebM7r7Pec/ut+fZd5fHBo9+eWh5l8ulXfurfO3PyyUIgiAIgiAIgiAIgiAI
giAIgiAIgiAIgiAIgiAIgiAIgiAIgiAIgiAIgiAIgiAIgiAIgiAIgiAIgiAI
wv+NueePt/+vaxAE4bbghA/cX1p/m1qa9L8g3KVoUXtyrve35crOV/W5VtL3
WKn+EL/rBI73H5frkU9MzPjwuk4PGZgmPiMI/y72+LxDqr7SfZt2Rd9bbvS9
9nn7M6pxxsX6p0vrSyu32Q84vrNNoSf9qy879gnmtwt+9UTvPNXZxHlEVvxH
6xRjU+iP4j/C3YYTPSNbdV879efswnP31Lfoe2NE31vqX9Nc5oe+7RyUdT2a
mcM96n/78MndyHPmpEd6rWlOJuYPL+eZ/v5F8B+nXq5y/fO3DModj+t46WyM
+Idwp6H59cvD8zGsGZ6PRmS7k9iuu5rP+zZXGRf2Plvqc/7IQ2Hok1PuERg3
NCUWcXqNDOxPGleq/mbo8UlboX/J9sw/WhanQz9vpEfvH5pf/9Xwn/utrH/S
v/rBhuul/4U7DT3U9wju79VH0V968nOIVsaj6FfHP499t7FXUan3r5eNdYST
/Fs8nrf7N+J93dnuA72hl5Sp/+1t0yMwru2SaNQR3uFj1De6Senz3yzfpMRk
1LF5Vob0nyD8GaN6NTz/7aE6+tNctwjRXr+V/T+uHvpOv7pN3X91/N/D+A0x
BejTcgnod9vnjRu+8Rjy6EF7y9S/dofGn6Pvm3U5hTwDJzFP8O9l8g+z0U4L
8/Vuie+SRrMgfL/UnN/L9v1gV+s3cR3MSfAdJ64Q3zfM3O/KpNfCuwdj/uqN
A6BrNB96Z9VvBbfiP/qQiYtR94On+N0hwudLj/zri2+wLjNGBr8t/if8Fe3A
q+zP5KroT7NtnyLeL6ns1zM1i9GHYwLdyu//C924Px13KPJoMWH0Ea+vGKMC
6B/bRxar9NYnPfB+bi0d9DPydM3nemGEBb1xxWYMKlL6h3Y4nd8H5mw+gHnC
LzsYV5XrF8uaTj+zk5X+YX7m3olxAVuQx06pjf9XONHH6TuFkdQvXqnWd/CC
Xxm9YrbjeOYbqMO+6KYPutqyjjavqf0ryWczju94JgXXq8/lvZi/gS/8T3+q
F+dPbaXU2wMK38e48q+uQhzVYR3XTWO/xnb3Nb9Af+9F5XdeJ20L3m/s05ew
bnPtSkAe7dJBrJv0Jm2/Rb5Zj9gqvdEpcRnqDcmPw/gKOp4HZoVCRGdfrc38
/ZMSlddvWWYk8sd6x3Hd5z0L47N/nos8G/zW0Ud9lqv0+tGsFai3wYqV+B1a
dUllPfsQ7UJzLZ9vKxKU1//7yrVxPMeN47Z3IuvtGbsF4wdMwXubc/6Y2j9j
EwdgfPo8nJ+W8TDq0LcOwXPIvCcoCbrseRF3ov/aARHF7J/V6G9n8gRE+4Ff
EPVz3c7hPAJbnlf276gg9rV/FH1i2KPUlR+EqE3eze3qe88pf//Zl+k3V+tA
7/wYgGjsyGRdgZOhN2cEqvUz3fStFgWM+ypTP20263m2iNsdOyr9yww5yvmL
MukvT1xE1GqY0Jk5ecxTq67a/5K+of7Z3Yh28q/0uxNTeV1a76X+nq+V/mfe
R5/VP0ph/eXiOH9eLqI1vzn1nYKVeltfVUS/SuD83eKpi2DURhzkdk6e0j+t
52eyft9IxohpzOcfxe0Xw7jdeal6/ZY/hvNkj+bxISE8H69Q7q8ylvMnmer5
0wZzfEV/xvnDeB6LGK3c4cxzIUKp1xe/wnETXmC9w7leNeo/yf3DunP77Nib
nH8rzlu74Y3zrM7fYXRN7u/Znvsr+annb9qa83SpRV1/jdtZrMNJaMH9u/t5
9P56u3Gej2FfNZiL/rauxDJ+1+QCzr9SIKKR4V2ivH8b+UNvpRQxlpSD3uyz
mHkO/YCoJx+8oOyfp2tAp/fvhejUiEW0K//E/d7LoXemLFDqXRsKbvjD6/Qp
/07ME7YSUTtQhfpNfmr/8j1MfZ0KrP9STeYp7sK4J53XZ+1+tf8U7oHe2LOf
vtmU9TiPV8R44+HBzBs7U6m322/l+JBPWUfWIeaJPs79mwxej949lHpr/HqM
s/I3Ut9uHf122wb6b6PT3L6vROlfVoM11J/keFfzVM678gPq11CvBZ9Q6o01
cay3eTzzhCZR/y7z2V8yj8vcqNS7Ds/mOJ9o1pm2gnnutbjftYr5lsQr9XbM
jedVvSnUHVvIPFeW8nqUMNrH31Hrqw3lfmsUx/edyt8zfRbzdY6iPnKqUu8s
CKbuwluc9zTn0YfwPMxMzu9Knas+f0EQhFvhD7/sn04=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 75->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
zOPwOzkKj6s=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 76->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmnlQF2UYx394oINFmBGe49p4joLiVZrHijZOmhbehMSKgHhiNohXuogo
ImSIAnI4q5DgiaFoYIMrgkoUY4gCirhqaF54cWiHkzxf+6d5Q4GcGns+M8wz
+3vf7/O8u7/f99l3Vzu4eY/zaGgymaQnf1ZP/pqaGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIb5
n6EnLj879N9eBMMwLwRtskV6Tf5W9ra/yP5nmJcTrbR5abW/tVnNrot8rh7u
VliT/w2vZkb1uJoSf6k+fUJN37+b+wzD/LPoWSt+FPp6sccy8n2DsNvk35XW
d0TzNEvLuzX60qv8XPW4Uf5OzfP+BrWw5ACt42jy7fr4X+qaeaw+ev3U6e+4
/zAvHRcLLgt/1/v8yS9q2yvke93Op1b+1fQin+r5UsrrVyg+7lcn/2tfjM8i
fWqVsP88Cyn1XBHpzDvWTd/D7iD1n1sflrH/mZcNyanXz+TzaxYldJ8Oj6N+
oBd8Ar8UuJJvtSjbe7X5/SuV8zdSvlyr05RvcLda6f9ELXM4TOsbOaVOevmt
d6m+0mlYnfqPnhQ8j87/Uli99h8M85+keeEF8mfedPjDbz1Fo0sSRX1tE/Kd
PLvJfdHvX18wcNORJ1GTHfCeID93a/WxyamC9g/K1Cr0j1knn8u/kufJ6VQv
beJcqr9gB/nflOQtrP8sFH+bFNo/bBtcVB//amXtZ7P/mZcNqXc4fD57GPlT
c3ShqLeOhe+H9yDfKcneD4T+b3MplHQZm2h/LHd1oHzybhPpVcun923byhr9
q/i85ke6EwOOkG5yf9qXKB+1Ir0R1PG5/K9nhkSR3xssKqZ8Ls7XKE9pi+e6
/6urb8VV9y81eEUq6SZE3KL12D3n80OEBfU/5VvbSNJtOX2T1mOu1ur5QS40
W0PXtbE79Wd1dW6d3l/ofVQnymOfuYT7F/NXlJPh5C8lexL5yxgxCD5zTIfv
HxZR1GeuKxf9fuTpl+j9vvYoDT532Q6/9uqJfKlOyHd2sbB/yPlp9NxhKllG
+wep2SD0oX7Io21+gNjipnj/Udo9m3y++cwZqpc3nt43aKEz0b+moZ/JTZOE
+w/jeF4mjce50Ps9RfH6ifzWZxT1C2nJp9DZ+In3L40cqV9J1juPUh7PQvr/
EMohZ/TVyBnoO2sdhXrd5gbtT1S3/ojlgbkU+0fgucy/BPunvjZCvRKuJ9Dn
bcdup/o5v+2hui43KI8efpb6jxGSItz/6Ga9Sad/mbGN4rxFdGx6M30/5buZ
dYo+D8o8Kvz+0m9H0/pcEyiaHt5DHm8pnuombqD3J2qv5B1CvcfVzfR5mBlF
qdXdQLp+LUtDSDen8y5ah6VXrEiv5lQmUZ05Penfh4wh5lRPHp9IUeoUT9dH
ji/bJtJrUZe/xj53xz58z6GHqP6ACfR9yBlzse5pcQuF33/Wq3Se6ooLdL2k
vAcxpJdTt9B6UsJo/VrlaLH+36b7MfKl4uQFfz84QFFu8g1Ffd2QChq3cKwU
fn++LtA3jERcM4h0Wp8RFJWz3yP6BVaI9JKFGemMbHvow90oyq5XKeqB9sjn
5yDUK4vKqC9II6ooahNHQm+djPhKOaLRWti/jDfOoE9ZXaGoOlTg+PcArGfz
QYrq55XC/qUMhF7r8gN0w/LRpxb1Rd1cjaIUECXUGxFHUFc+in6blgJ9v7sU
5avv4Tr0Gyeu77MH+uQkXIfzCejX9rHQpxUjX/8i8f7JPRx1V0VjfCiOTdEb
cV4xwcijJAj1ivlajMesR93MIOi8Q/C5ijySQ4xQb1j44rq1Ww59Po4lBVEL
DoRueJBQr+ZPRX9vNAP1HBTMy3KH3h151KIQcf2xrUkvTR6N+6DZ+5iXPQr6
ROTRxywUX7+D0GmnRqO+31jEnROgH+iK88tcVqfn1xeNGptIvtJT+pC/JU8v
xMaNq2j9Q50Qi1s/FPp/5Hj4MuAQ4q4yxC49KY86KA/5Pi6pEuofNqf5WiMZ
feZXD4pS0QmKRoUT+k6Cs1Cvx90hX0s+7aGzsqWops/BOvKRR19vLexf+uUS
9KfzjymqV1pCVzAcsUM0RdntgrD/mB4Xkc7IeYQ+NaYcebZiHXrGbOis4sT9
L+c41j8pC/327Rz0387Iq4/ogfOyGyfUG6H7MD93L/JM2YH1ZCCPHog8Urvr
wv6nJW/HultAp6btQr6v4jG/zYmn+QuEemVsGOr4h2P8fhTquyIqnkEYn7lV
qJesAzE+IwDzo0NwfB77TdV8A+L8SLF+1VLU812McctVON8PEOX7yKMXbxTq
1SG47+ntnKFbOg31fnHH/J4LsK5JAeLzX+eNOiGemGf+Ga7nZF/ktVuJ8dRg
oZ5hGKZW/AEakIHz
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 77->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmQlMFVcUhp9WqxJxqbZaxHZccNcY7eKTRofgbrUqRGuoOAG1EkXQViqC
ZpSiVAVUFJ6AMGAAK1oFtHGpZURqcY/UDXzIIFo3cEHQNqJWzm8T29xqhTaS
9nwJ+XNn7n/vucOcM3fmtfPwGTf1FZPJJD3+a/b4r6GJYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YZj/Gw9Dcge+7BgYhnkp6IP6FHL+M8x/E7X83MWq/FaS5l8W5bl6YtrpquNq
g9VBwjrgN89K/tJ+1prUCcXLvJbrDMP8s+jRLY9V5ZV8cOrSp/NLbzQymPK2
OLm0SvXjXW8K87+fOx2X6+thovPaAcs58g90vyU6L92a7l11XJNsI0TnZbuv
ttD46+JKapT/Z8u2cv1gmD8iBZVcEOZt/5tZVcelwR9Q3hpHfYT5q91386Pn
f517a54+L7Vt+iXlvWNgMR2P/1VYP3RzyOJn5aW8acQBqkN9soX+5yFvX36K
/F1bVMuvtm40jq6Db/iNmtQPrdf8OK4/TG3DmOZL+3ppaCnlifRpKdUDOXU7
3e/GcJ3yXl44/vaz7l/D3lhG/TMnq5RvaT1Cqe3/2knUh0ph/XgealjF97Q/
6NHimfP/FbJdDL2faBdHVssv7fkpgdZj71+t+A2jhw9dvwWbJ3P+M7UNzT2b
vt9J4a50f+uBC0ml4+vw3J/yJuWN0bFBmXB/roT6ZT5WxeJwie5z/0VpVW3D
mkH7ByXyBsbVK6qXf0uzdaofnrOF8z8PoyKO4jAZedX6TqnFedL7ib6j07c1
yV+9f7E35z9T25A3J1NeqidXQSumIE9bhpLKwcmUd8qZnneE+e97jfb9crd6
2F9fa4N6kXAN+4a1YzBeej2h/3ck/ZMQek5vn0HPe1POoCs0b7ot+bXrJX+r
fige7qspX0eNoO+NxsyW2N/Yhb/Q81t3LdxB49TR6buD7H6q9EX8SvYcikMZ
7XKddHBstd4f9L4BtA61wGt/jeqP+fZMrj/Mn1GaH6K8ksxvUZ5Lzl6k8qrD
aFtsKG/V9RvKhe/v7RvQ+4Lmk4H8/CIT+RpTH89rnyRS1cFDmP/KhPnYf3hb
fiafix/8kRZSvagAdamTjbj+pF09SPlltT1D4yTb0u8V8uIorOv+MsQVq4nr
R6bbD1QnBmUdonmcgsiveV7CvqX5DNTBuYrQr2fb0P5EzgreR+Os2EvvG3JB
KvZRjo7wSwPE83fzpn2FmupAKr2eS99jTY7tr9K6Kt/F+rddENYv/UHPjXRe
LUsmTWxM3zlV89rjFP9FF6pfUpI1X+RXt71BPqVjRCL1P7KA2qaD+zOo7dzr
BOmBblkiv+FcFE3jV75PqnbA+5J2u/MG8hWG0Lq0lPSvhf+/zKAo6lc+Eb/v
rLkQQFq3Df2epOYcJJ9s3IgR+uOPfUPzlxyi6yDLTSlu9dUMUnnhLYrDVNEq
QXj/Xb9C/aTI3TSOcsmW6r48vSid4lo0gK6H7PaOp/D6zQ/AOq0p1F+yj4+k
+6BLuIWOnyqjuEypUVNrY/3VJxxGfj8Ipfw20hZA990kVes6VZDmjrorvH6d
lsCfFkYqXTaRT1rhjHphm4HxZiRViPyaayXqzZAn+4OdYzDexp2kWqgr4jg6
SuhX9KvkN7pDlaNN4N85nlSO3Y64fDsK65d6OA++h1ZofvGTcbrD1y+YVO/w
SFh/pPBTqG9xJ1Avf8yHet7EcZfe5FOiVgv9xrlM9HffQ6q33AL97hD8u25B
86YJ/bo5Df1PZkA3bCXVPsY4Jue9WM/MQuH7k9RhPfq7xGOe/Qlo20L10CXQ
ogyhX20YjvPWNfB7RKLtE4X2pAj8fx3ihH4lywn9+/hi3i7wK24W+HvBp66w
CP3aDEccTw/BPF6fQzMxr3E2FucXxIrnd7PiOTHEG76xExHP6JXo3yQGcUWH
C/1G40l4TkUMxHzbhuL/eWca+kfPwrgPgqr1/vpvo+7OprzSUiXkt9NKUj3s
DKn63of3aF3lrX4Rrj9nAvIycAep7GED/+J2pMrZo9BL5+8J7x/lbfIp4YNI
pYDPMI7PHlLdFnHJi4cL/dqJUtSXsb3gH9Yb9aqPBXGlYn2muXbi+uVRgPoS
/4hUs7dH/3nDMf+mWFyf63nC+mMynyaf7niXVN5yB+2RWI9a5g7fgRRx/VNz
UJf8oZJSCD2NuJQ5o+HLGS70G4Vp6B8B1c5vxvwhR7Aucx7OV14R1799GxG3
ORXXcdEmaLMU+Cvz0R5cKPTL2lqMX2BBvAnR8Lmvx/H+iTjeNUk8v89s9N+1
FGpZgfjtw+BTItFOiBX7+5kxT/4w+C8EoD0lENej7zIcz18j9mejrusN2+I6
OI3Eeld6QW384N+6VLz/bd0Z5z9CHFruLMRbNgeqLsI6EpcL/QzDMC/EbyQb
pq4=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 78->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
8qXeGxatfv6CIAjXw28vVo9B
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 79->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmXlwTmcUxj+Doe2IpiVJlfTaKrZKUSKk7hhbgg4R1H4RsWQhopWEyiUS
EYJEUIrcTxD7ErGEqdxILBVbBCWLXFt81jYpobRUzuOP/vHKEFqmPb+ZzJn7
3vc577mv7zz3/T61h49zH1neZDJJT//ef/pX2cQwDMMwDMMwDMMwDMMwDMMw
DMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwzP8M
Nbjp4fZvugiGYd4IstPKHO5/hvlvYqy9YJT0t2Tjf1HU59oT6WRp/W+0b3yM
9BEhma/iE1KTrgr7DMO8XoxPQ/eI+krqPXlxybh+M+UO3c9J/UXYfw89fi2t
L/UxWXQ+UPwiS533PFTnuK0pJfVUuned+59hXi/atdxLwr46l6TTe/uwQn0r
BbsI+1ceGH9QNK63cYspGTfO171aErXDg4V6dWitlNL6WnHYh/x1ncrkH0bC
HNKrQbstZdHr4e5BVH++VPBK/jNrqDnlVfQM8w8gO4ymz7XiNPgs9cmNNXTe
N3y98d7fdov6zpj5RWGp5/wzxxL+/vk2OqfFUt7A25RXPv1Bqfrnofj0SqM8
1cPLdn5I9M6mc8zeOWVaX/ZstRf7MbxM65tmVVlK+1rxQAyfX5i3jtEW+l6v
Wj2gz7dScJWimrYN730P9K0U8EjYP5pzrJnmN3xM73n5vTydzuvWF6hvjUJf
0smm9FL7T7XrEkz3XeOHUb7k6pGkX6Ok0/phT8rUv4bnpHTU0fNGWfRS29QF
5B8J8ftf6feLgpAd3P/M24byYQDe7xMnUX9pEf7o9+OrKarnfYro8/9D7d+E
5/x5LutpvuMG+n1Aa55N+TRTHvI64Nxg+FYW6hWnGaHkO14Z1O+6/eZ9NO/Y
djqvG5mF8KGc8kUv0j/awIbjaX64OZfqGNYd55uZD17IP7T58/1Jt+Nr6lft
M7tbVEee8UJ6JcdvKs0LnhZG60bfJd8x6p+8/TL9L5knziDfzPCj30+kG03L
5D/6l90Gse8wz0Ptb0t9pXWzoiipIynKlpQivNfrUd+qw+PvCr//j9l8hfRW
2fCNg2b0SdUj8A/rCfCPKROF/S9lTcf3DXd8P1ceOsMv7PsiT0YMfCn5HaHe
yBqXQet+UvM81W27kM4hpoeNcO6oOhJ5+qtC/5ATven3AeWP00dp3clzSS8l
P8Q5qJwH6smcJex/Y9glHftTRP2pN2hwjq4XzYR+gYx9CBkrPj+1XLeLxgfU
2k3rbhhA/59iBHwM3/ioDXQV6gj1UsU762jdZacoGqkR2+i5bfIoj1SvCvmO
otzKFT7/74cSqD4n/9WUJ2Ia+bnWyHonzT9U4RT2tUGaUO9ts4LGr3pomGe/
hvLl/0h55Sl96bn0Fh03CvXyziW0Xu2jFE2xYyOoXnuXKNKt7wrdkIkrhO+f
tnlbaJ2jHrSenuy+nebdKJ9EdXy/eBXF5cVmkV4d347mmZIubaD9S6xGvi9l
FSfS9YFVtC+my5Fewv2P+oryGhndMb9FhziqY+5G7MeIs7SfqmXp+LfRh7Vv
hlJfK5emUdSvbKVo6JcxHup5j+pv36pYuH9RXtSX+rklFKUTXaCPaEVRWpFP
UQ6Iuyf897tgIp3SzBbxdhD6POA4fEeToO/mKtQrhgX+ciKfovHsnKJ7R0Hv
m07RGGQl9C+j8Vn4giOi6vcsT0xP6CdEI4bmCv1H/u4U/LJTJnStb1FUnFvS
fDlpCPbFJe4555998N/H8FtT4C7k634FeR79ifu2I4R6df8GrFsHUYlbCx+3
24446xDGY++Kz0/W3tDXXIR5Sgz2ofYM6GskYH/PJAr1WnQY5s9YgDwdonAd
GIvrXuG4fjdevL78LdbJxTy9+XjM7xyJ8QLk0WosFusLz8OfV41B/VIv6K2Q
V3PDc+jLooV6qccO+OrxnygqTVtj/jUv1O/qjjqCponXD8uFv1/uAZ1LV8T6
nogOvtD3m/pC59d/G8kxm/pKCfKh/la3LkWfVyuiKHdse5/iuVoPhP3bbRD0
wRaKcr9cinqMP/SfH6Vo2Oy/L9QXtUBfT3FDHseF0FfaRNFwG0F63eIl1Muz
LfCpZvY0X/NqCL9qMQ/XPmuQ546d0L+MctnwhT7X4FfmJniO/GbQb1mO6Pyz
0H9MozJJp/VBHXoI8kk3bZCnXHPoli8U6rV6afDLAanI0zAHfpeNvCb/1tiX
1E5Cvbx5M/TlN2H962spqgV7EAMycL/dXaH/SVPNuH9kJdZtuwLz0uPwHHeO
YNzljFCvLJiH+4OjUf86RLkgBvW4L0Ud1dcL9abR03H/bCR0XtAZxbEY370c
/76Oq8Trx4dhnvNs3L+4GHX3XAbd2ATkDU8Qrx/UG3W3GYV59/yQbzLqMi7O
x7iDJtTrrp5Yr8FIPO++SYgnJmN+4Jxn+7NIvD7DMMzL8BeYbZtS
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 80->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 81->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmXtQFmUUxj8RFZ00UsrUtMW8geYlzcYrW17G0hDBrxS8rOYNbxB4y9BW
ZaRQ00CRDGQBBRRB0RDJC5tgXkaB0pBAbRVFUUQS8a7ld57+aJo3RGpGqvOb
Yc7su+9z3rO733n2XbUf7+U6sabJZJIe/dk++rMxMQzDMAzDMAzDMAzDMAzD
MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzz
P8NwnrfH6WkXwTDMU0H1CMjn/meY/ybStXZ5lv5WHfbniPrcKKx7uDL9r5lr
pLFPMMy/hOZeG6jvHzYtpTg885qw/80Tr1bU17LjXPIPU9Qkob6yaA/HFLJ/
MMw/i/yW41lRX8lpXrpl3Fj1LvW/cX5sqWie+lrjH4X6rcXrLeNKVDb1rTEv
QKiXbW0PVdTX0var6ZbzevSJKvmH5FcjifQPOpyvil5f2mpT2qOoThxy+m/5
Tw23QPYvprohWx+/QP1hnUD7eyl1Cf3OlXdal9D4rAjqW6Vpl18q/P2+eqeN
pU/kBsfMlqjHZoTQvqH95J9I51ujYv1foPfb/C35x6/NhP7xWL1N+imqv1tw
lfRGiXaArsNqZElV9KrLrFjSNYpbz/3PVDekDnMKqD+culF/SPdOYL9vdxnv
/XYdqG/1I/nC/pXincNJt7iI3q/KNYdsS/8rLk703la7NyWd8ll25fp/3ns+
f5wnqSHUf5pzSpX8Q3aoT3opyKtKelNMkIvlerSoZyrcpzwOo/2ldO5/prqh
9J2K/q41AP1xbgaO62kUpUGvlFEf771/Xdj/DXvQe17rkQX/sF2A/ULuYfhH
ihl5Uh3LKtznF9msojwxvei7Q2u2oYj0afAjOa+sUv1rTB4cROudNWO/XtIC
3x9edyqll8ODE2j/Mn9pCuma5xRTPWFnK+cf9ovXWfRSbs9Qqnvdiiuo3/aJ
vl+MQodt5KPhXWn/ogcV7K+Sf8Q1CCDd2f2L2H+YP6NN7UF9rfWZQ1HN+pKi
7GlgPLgz+laKuSH8/l/mTf1halCb5uveJdQnWkFX5EsNpqj4+Qv7X27udYb6
7Fgg9bsak4l9uvMC9FtqLvYPR8qF/iNHDz5K/V77O/p3RuX7gRepjk+SkGeL
L+pJWi/UG4HKQdInt8wi/epvSK926g3/mjEKdey2FupNSVvo+0SNCM+gPHus
6P9Dpdt9sY+q5Q7/KvUV+oc+N2sXrZuwK5XO72j2A617v4x8Q2k8m/S6X22h
Xp2+KJ7y27lvId3wETto/XX9KY/kUxffLbn3zgjrv//CJpo3QaXvFL2gSQLV
XRgJ/5sST3nUFg4Zwvt3OSqCdIM7RlHMORJHMSp5M0WnsXRd+hTHROH6B0aF
4Pk6htE6Q3zpPWDod8nH1bYXqB5tZV9NqLe6gfqtHkTTPH/rr+l+XMqlqM0f
HUOxnzlaeP/s8qlO1TNlI+l0FTqzN91HNTOZ9Pq8Fj7C62+trqF6f721lWLB
+Eial7iD7oe2eQ89H+XAz7Oro//qoXeoL+XiZdTfxsUUipL7OYpylwnldP3D
ht0UXn/pBPR16xSKRnwv0mkvuVHUx2RQVF0Ty4XX39OKdEqiUxn6JxR5fIop
aqoT6jFPFer1m4Xwq8lF8Cu9N/YrrluQNysP0f6B0L9Ma0/Cpzqdpihdu4s8
I5dD57YT9yf8qFAvnTwK/bhD0Jnu4fhDD9TvAt8zGmUI/c/Q4bemlGToIlPg
lx7PQhfaFvHl4WL/bBUD312+EfrxsYgfJyKeP4V8x8+J/as//NlotRLnS5ch
JgZh3H4Vjm9uEuqVgA8wnq/i+lvNx/MYhGNlII7VtEjx+rI7xn18Md/6IzyH
btBpfv44/3mo2P+fw/pyE+jkLET16BLM7xyIPM4hQr26ejzGgwdj3cLRuJ8L
xiHfGzNxPsBLvP/d7or8b76N+9UTesn8e113PHHcMEB8/U8ZqXAm9bW6expF
yS+Oot7zEkVDmXOLzl+2uy2sPwB9qYado6jssiadluYCv0g6TVGueeyW8Pe/
qDPptI6uyBMQhD7fvw/H0wegHv/pQr18+gJ86vqLNF+60oeikesP33I7iHwj
EoT+pXrmwa+GFlNU6nRAHu9+5fC/Nch3ME/oP0oifEHOPA6/Ky6B362tjzzD
pqGOTyPF/lVvL3zFU4f/Ds1GHaPyMf58J9zXpROEem3hJsyfG4/ovQ3RZTf0
28vgn7PuC/3LiI3A+TrR0Lljn6fNSkAcgetSnO4J9bL9FzjfJwT1T1oL3Ymv
kPf1nThO3y72X7vFOB+zHPOLkE/eF4p4KBLjbTaL12/ph/Vz5iJP15U47h4M
fe8w1DUkWqy38cC8fgrW2TsHzzF9IZ5jx0AcnwoW1x87CevthF5Nn495t1WM
712B+P4asZ5hGOZJ+A1woY3b
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 82->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmQlQFmUYxz8HUWdINHMksWSxPLsUzzGVBXOyRDRzPMpjVcQ8QCukwGsl
NQMF8QBEkgXEC0XEW1AXFJS8kGQQ1NwEBcQDxFtH83v+NNM476hAlk3Pb4Z5
5t3v/T/P++7u8/92P+xHTx4w1sJkMkmP/+o//qtjYhiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRjmf4L+oJ3h+G8vgmGYF4J+/2g29Xf/cZnCPvfvnf48/a82OLiRfYJh/hvo
t2+sNfer+naNUnPUNs67JupfxXF+ydP6WurrlkOfR4deqU7/S5bdL7B/MMzf
i9TC+TdRX0lF+br5uHFjEPW/WturVDhvyZ0c0XGtdc9o83F9+qgiiq06CfWq
e7cjT+trRXZKpXXc6Sv0n2chH+2xivxr+/lqvadIM6f8Wh29ka357KuGnmFe
BEbW4ELzfS0PCjtJfVZkf5r6JSeWvq/VxBD07aSWZZW5/9Xv2mvm+YpjHuWT
o+pUSv8n8qCyFPKPfe2E/vFMvfeYs2ad1NW5Sno1ZlkanY+hEVXSS8WnYmn9
EQ3j+PmFedmQojfn0/0dNIPub2NTKkW9VyZFKdCW+lbrdl7Yv7rdEPqeV2wC
CmjeuydPmL/nDE9lP3Qn8f5weNpT+19u7eT+18/VPcPn03pyrQ9RXHCwSv6h
jnQmvVLbr2r+86qDo3k/8oG6x6vTv2qT0DTuf+Zlw3jUl/rCOOqBODEGfbI/
FXGmQzn5wELLcmH/20SvIN3ozfCNazVJp7+5G30f5AL/WFF+Xah3jP2GnjMW
/LCY6ngd2Uv9WjekmJ4bnKKQN6GOUP8k2vJGCynfbPczpO/S4yKNU+ZWrv8v
pW6mfTVbd5nyJPSqlF4Z6OtH8+MuXaL6NcdX6f1F3+SdR+elyajUKvmfbc4Y
0t1w8mL/YZ5E7/4+9ZXs6klRX+1NUbuVjX77yIn6Xuu4/Ybo/jHS4/Ge0Gcr
/OOLTIpyu4eI9xORr66/0D+MvJHn8Hz/NfWJNHYk6dR3fKAvSacotX9FqNcW
ldDvB/LyGHrPMDKi8D5T4Ib1ZISjbztuEvqHknz+IM3vmHaM1vFoHem15Ot4
3h82HP5l+b3Yv/pZUl9qx/wOUL2omDysJwO+VdqW9ErWJKF/qK4Dd9HxAP/d
pFuTdYLihJbkO3pJFn5/6dRYqJcD79F7hZpsRf9fkS+WbKF5Z0xZVFf3v0p5
rjYV//7R4Rz9zmtKtl1D57nbmg00f0jwDoqz7lIe1bTigLB+UrOV9PkbVlEU
u4yhPKYOceuofpk97U9JsokXXv/cFsFUZ/mjMNKntaXnPlUKCKQYeIzWY3Q7
HCmsr+XEUH63NlRfckhPJN2SYDoP8gcd8P51aE+08PrFpNBxY9lrFDVrb9JL
n7hQ1O12kF4uLZskXL/rPTynNm1I519RrCNoPEuKxHWzWk/H+7Wa8jL6r1rg
R32tRmygqDfLpii1trpJ+3h1DkUp4+NbwvVbzcbzQZdiinorV9JrGyYgn/UV
ikot/abw+tlZkM6wt6OoOE+nqMZlIt+nFfnquQn1ysRi+JajQVE9WBe6roMp
yoFH4V9H8oX+pQ7LhT9FIOqvn6RoSFiX2hS+pTcfKNTrV7dCn3aIouRxHHn6
wD9l67ewvyO7hP6lhu9EPYtdmP8zxnLRHuSNwrokvy+FeqXmWsw7tRr774yx
fGIbxvV3I8bcEvtXoi8dV+aF4fP0EKw/MQI6+wQcj94o1EutPKDfGYz5oxah
/ohQrMtxMfJlrhL7b+5c6DosxTzXIOQLWYbjCViHHBUp1GsZS1CnfiB0DZBP
skEekwvySFPCxPVdUFfquQbx7GzUG4k88rqp0M3zFepl3wm4fvXGYb2Tv4Iu
8luM4/G9ofwY8FzPr/80Si1b6mul0AP9vSMe4+H5FPWgmbdpvPLDO8Lz7zKW
+lJ1L6Koz7xHUYnog3wPUilKFr/fFt7/5Z2g9/+MouYXiT4fdQbjJOQxCqcJ
9dIvxfCZ9q1Rv0VP6KXl8K1tWfCxGoq4fps86O9aYF5zB+hdpiJuj8P6HKKF
/qdePEx6ecAF+MO5O8h3rTPWP2wa8jZJEfqX2jypwlf2Ir5XCD9uXIboMxT1
vcT+J7msp3lG/zisY0g8YkkGfPf0TYw9bMX1C1Zi3vUo+P7DaIwLkM/wwb50
v8tC/5PtgirO/zLMnxGCevdXVIy3YB+J4udHfcBP0Ecswrz4xYix4cijrYL/
O28Q18+dBb3NdMybEwi941Lsx1+ruC6xQr2ps4L9nnBHnt7IJ00KQL5s7M+0
K1yoVzw9sc5G46AbD702Yh7yhQdj/HmouD7DMExl+AOKFowg
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 83->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
BqGeYRjmH/EHeL6VOQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 84->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmXtQT2kYx38uU+7SLnZdtoPZapZZt7BhpxPGVJrdRO7LWSl2XXIX0h5E
kevS1e1EqZDLukt1UFa2lV0MiRxlkaVCLq21rZ7v/uGPV2ztDsPzmWmeOee8
3+d93p/zfM97jhYjfdy9qplMJunZn8WzvxomhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmHe
EeTE7tkOr7sIhmH+F9Q8q9Nl/W3UO7VD1Ofa3OFHuP8Z5i3DdyT1u9S8d0FZ
VP5wLRT1uWpmc7O8/lePjM8suy47pOdXxieMRuNy2WcY5r9Fmrz6grCvHj6g
57r+1LGI+t92WJFonHK5TY7ovFGrQQztG27Vv0XX+/YU6k3qol/L62vdrmli
2XV15aiCyvS/XGJ+vjJ61dpGZ/9h3jbUSe/T81tfZJlOfda2Gb3Pq52871Df
T1lDfatfM79b7v0/1XYGPeeHX/KjfYP3/TjKExZ7pSxqY5LE/f+y+jK7JFEd
nusrpDciPzdI7zG8QnqtT+eTtB7798pf/wtQ8vE76D0tY9g/mDeObpt+o/4s
1tHnk05TNKan4bnf2B73fcPfxff/4DrxNL7kSR6NbxRwLuVZVD0uHaXx/ltJ
p52eWW7/6KVWs+m6S13358dpxyNOUP+VrqlQ/8mNk8jXpGbNKqSXml4PJR9b
qp+t1PtLD8cM7n/mTcPInk59oQ8ZRVH28qJoKDvRL63v3KP+/fTyPdH9K9cc
OI909rPIL9SFZtDnW1KUpkxB/qBTQr3J2q03jQvsH0HX7XPoea+bV6f3Bjm+
LuqqV/pK/auoDhNJ77yP9jHy4NTrpLvg+Ep6eeu0IWX+pU1wwnM7QLtNvvbt
rVfbP/hPtkh5/riGROvQk5cIv5+8CMmt/tiyPLL3Kno/U81KKvX+oYb4uKa8
fBjzjqGM6EN9qawai/78OIKi9PgI+j5z1H26/xp6Fwu//1fpSX2ltDfQX8Fp
6P+H1aDvsJ6imrX2vnj/4E7vB3rbFtRnJtuvoXf0g28E/oi87a4K/UMrGHSK
5vEIvUx1pLWg74yGOgT7jiqrKaqeKUK9fi6d9hemRx70/xz63AR6HzJc+mLe
a75YX287oV79aBftc7TaqWk0zulP8h0lGn5h+MNPJTc/sf+MdzlE1x0K6TuH
nJdB30OMXi3p/UuNzqE88owIsX5CwTa6bj1tO8URC/dSrDqL8khp5tjXDRtw
VejfY+1o/yZfDI6leav0S6D1LL54gM4vSDhD+gP2x0V6w8ptPdVZIkXRfE6B
lEda0XYL6bJsaX3youo7hfUXDV0Bf29C/q9Hui6m2KXmSqoj6xLVo9tYRgnr
d7HcRPMnHttI8448uJvyjdH20Hnz/qhnXqto4b+/Z6hG55+Ukl4OT/6Bxncp
pqj/bE56uaX1DKG+i7qArmfVpzqVlcURuF9Gr6M8IR/S72AUuE1+I/d/GztS
X0sW6yjqU89SlGOrPqC6ty2jaIrxfSiqX9KmUl8r6RfhE+3gE0oPZ/jFmFzk
D8h+INIrPWvAF8KdkMfYhjxWJtJpXzhQVLOHCvWynk19qTcspGiYIY/eVEO+
jLvIP91VqFeXXoK+Vi580Ooh+ny3O+mk60mI04qE/mfyPANd4WnMX3oF0fcT
zFtrIqJPvtD/5AVboA9JhM4rGf5bgnr0WQ0w/ylVqJcORUI3bit0TXYhn/Uh
nI/NhA8XdhTqleJ18OfwGIzLiaYoN0dUO8fjuEGq0P+MoDDM1zUK9SbB77Xi
zdC13YBj5/1ivetSnG+/FuOXrkGe/dDpXZFHqpYg1EtFPlin/3LU+1kgdJH/
1JWM9ekbYsTzDxqA/Dmzoa85H/lyodf7BUDXLUS8f13liHFf+kOXOhPr8AzG
us8EIe+yMLH+NaMG9qG+ljOD0N/bEyiq5+s+ovXc2EBRq9PrsfD+UVX4RG4r
0mm3a1BUbkxCNKsH/TGTUC/72ZFeC/uKouocTFFenUlRGfUN6jo89ZHw95t8
B32ZZ0Pj9azuqMdmLvJ8cBV5Z0wS6nVfA3r/p/Cpg61QT4eB8AvfHRQlh0dC
/1P3/QK/K7oC/1ySD7+KtsD8rUejnu9OCv1HijuC8dURpeln4L+1byLadoJ+
TphQb7LcCp9M3IL5NyWgnjlHKRo74b9aG1ux/4ZHYV4nRHVsNGKjOJy/eQJ5
Y58I/U/OC0Gdh1djvgREZXc45rWIQZ5xe8X+eSwS1+OXYd6s5Tiu9z30nfBc
UvPjxPObbca4RStw3SoQxw+CUEd/1GP6aaNQr8dj3aYW0Jmuzsd6PWaijvMB
mD8jVKzfE4b5fFRc/8sP+mY41vcFQ58UIV4/wzDMv+Fvzw93dw==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 85->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHlMFVcUxp/QRaWJ2hqwtdpRaTQqRUO01dIyxooatdFqUikKQytUQxcF
pLjhGEVQ0NaFByjq4wEqFgHBIlqFkQJpwahsisbA4I4Lm6KogVbORxP/uKJi
G0l7fgk5uTP3O/fcYc4386bfl99/5mlpMBikh3/dH/51NjAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzAM8z9Bv19d5vSii2AY5l9BtZub21Z/m/qHZ3L/M8x/C63f5JSWvlZK
Ym+2RN1UUy3qc1PvhZfb6n9lR1Q+5WnwufI8PqFdjdbZZxjmH2bd0lJRX2nN
U36jvtWda+n88Km1onnqsfIKoV4ZGdtyXN7a5Tr5x5LhQr1kzitpq6/l5dqv
lKexm9B/nkjn20G0vlQk3OeTUGvvb2jRaTaFCc/jP3p02Cb2L6ajYfr9WFXL
fSmVOeTQfT7m4zMUm7rSc9/km0N9q/r1qGvr/lX9fbyyHj0QU7uPdC7u9NzW
TuwV9v+T0EJWU11KVUi79KqnXEm6hUr71s8p0Oj6WFm3S28IvLeX/Kf7diP3
P9PRULWqS9Snk2/S/S2/Uof73PEa+n7Ve+j7zCJh/8u27ol0f7tZXyS9y5iy
Fh/Qwy2ob+WXX0e+o7Pa9A95aFbAo+e1pv4RpN9ylX4/GMzftKl/HNIW1wKq
Tz7Vrv6VU+fFUB2LR5x6nv41zWo6wf3PdDQ0uznUV1raRoom57UUdftCispr
XrdoXHS+XnT/KmHz/Kk/8vtC73uf+kw/EIS4fDqOV6QI9X8jr15Ez0fZcdcR
WtejzzUad1aRp7b6qfpfDp02v8V/TO5fnaPndtpN+j6hr8p/Or3lhiTyry59
yNcMFg70HqTajn8qvVTcPZ7WbyrYTNfVJZr2YYh6tvcH7ajNLy15pE0f4n3M
2e5ou/zDL9qHfYd5HJKVE/WlHuFJUVHXYOx5mqKaoFD/KxEzbwvvI3s/+MeU
ZMSMe/CNIc6kl64cpKjp6beE3wkcJ9DvA8lhLPWZVv85/Md3GUW110XkqzEI
9er4P4+TPupIOenHzaF+02a7kU7yMVKU48T+JZkn/kHzE3oV0nrGJdA3RFK/
StaL4V+rvhXqTcM703cSLaA5j+ruNZrqkC3w/qScTKaopScI/UO1TjlEunjt
MM3f2FhM6w0KpO8demkq9IMXCPW64k2/LwyxMcn4ThGYTlFzoTzSNXy/0U8+
qBSu/+o8+q4hR2XtonnTfqB8mpXxII39bej7jHr3Tp5w/xa22+n49BF4T7oa
TXmk0O/2UMxPx/6q41KE6+/v8SMdbzodSftWwtbS+HQJfXcxNNomUT2h5hjh
/ec9xkzn68yI5zPTaN60vP1UT3gh1WPYujVOuH5F4g5ad6AX6SXXgFSaN66E
otZzAOmVY9WLhM+/6jdWkv64Hz0vJL9g2oduMSWadK46vhvNsfftiD6sbPGg
vtbOpFDUK8+gzyf0bMB4HUWtwnhHuP9mM94PBnclnZTyPvS5yyiqewopymHZ
DcL+C7EkvXz6A/T3F2HI90kdxrmoT6mYKtaPqoVv9b6FON0e+RaGQt+pHvls
PhLqtX3lpDPNvgS/KmlCnw+Z2VrXIYqS+0mh/0k9c+CX2QXQL69EtBsN3YAA
1OHzQOhfppGt70Xeh+G39oiS4w345oh3Sae5rRf7Z0Yy1h+0j6JsibG6di/0
NaU4fneSeH1vE/TrY3D9Qsy4Hrmm1nES9Nm5Qv/TJ0Zg3tmtyDN2M+KGjVh/
fixiWKpQr33aOi8gHLqvIzHvpZ9QRyfUpSf+LH5/nBkP/SnopL1GzLvRGp2w
DzVzp9i/nbdA39xafxDWVctUHB+L/ckXIoV6+U0PzFvpj3rLlyKWrUa+QDxP
pQRjm++/LwopcQr1tdolCPFcLvr8gtVdqtspiaK8f2Kj8P9XvAJ9NeMt0mke
Nsizx4+iIllCP9ZKqJcWTCK96YA7RfmVNMQZZ+E7jt7I183rrvD6u9VTX5pi
RsOvOo2HbvcexKV3EOtnCfXasEvwp6EWNE+qGob9xM9GHa4pOH65Tuh/pspS
+JzDdfjo3GbERfbIU7KSolJQKPavWUehT8iDbmAx/K5vFeKEUbg+wZFCvSFv
H+pPTIU+bT98uHc+/Mq1EfmDhwn1etBu+L53AuYFJmFchrxKRhHGdkOFevmd
bVjvxo7WfcSgjqx4rO+HPIb+h4T+qS0wYn4NomFhBPJUIK88A3nkhmSxfuca
HL++AnW+HYa6Q9dDtyYK94dLnNi/zy7Ces1+OD9oFfaTFwSdxzrkzYsS6+MC
sU62P/aREQz9ZNRhGB+O40nbxO/PDMMwz8JfA4uciA==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 86->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmXtQTnkYxw8izKxxl7W202yDGZda2zATw7HsknEbw8S2aw9tlMS6rGHs
cso9u13UVpKcekUXke7uRwmxzbJCbuskRGoXkdvEep9vf/62nWKGsc9npvk6
l+/z+/3Oe57vOe/LYca8iV7NJEmSX/21ffXXUmIYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
5n+Cmnfl/NC3PQmGYd4K2tfh+7j/Geb9Qk6ryLH2tekbU2VV5WPvv4R97h13
o77+N8qjj1uPy83vlb1OTihVa//knGGYN4tW5veHsK+WNz1q3a93iLhH/T/Q
4Z7wvEeppmi/fMl9K/V93ry7VlV7BAr9pmeHC/X1tRrXaz/VGb+26nX6X6uI
/70xftW7SHsTuaN+sX4F5xfzrqEsKKH+NN22G3R/BhWW0PbGa5XUt1F51Lf6
+Pb367t/9Zvhnoet/zD3ziX/iGWZVjVqfSkf1NQQcX78B+ap749ZfZrn8kb5
pW6LSyk/5kxqlF/uM+AIrb+yWb3r/zeM0Xa76XpckKK5/5l3DTNo9016vpcm
UX8Yk7fieb/3DqkWNprue/PQQeH9b65z20P93Xc9fQ9QJtpctOaAPiUnn3xB
tVRHtptWb/8YC38Ip/zQInxoHsWJQeRXEk7SPJa2aVT/SXYDi6iefWaj+t8Y
Jus0/vkNJa/Tv7J25DT3P/POkbKQ+koNKCCVgy2k2ryT6LdWiQ/o+Xfp7gPR
/at8smUzHU91RG5c19HvzW8gT7KGo57nEqHfGGcTTz4/dwvVSWh2iHzPx1fQ
dv591J36sEH9bxSfuUx1CpzKyX+grGH+4ZsTyBc2kr53GC8CGpY/LbJDyN/L
mdahdJ3WuPefqCDKHT26f15D/Prsl+503W8F470s91ZwY8bXPmqjcm69v8jT
vNCXG4OhpadJtWdnSNXvVlfj+T36oeg+UJ3Ooc+lrvBNaU+qVLUjNQPWQ7Nz
q4V++Rl9PzBjluP3Rdd+1GdKx2BSo3ovth+ZwvzQQ/Ppe70cmn6Nxl+8m/pN
lrsidzqF4v0l+bLQLz04Uoh8uU3PZ9l9IvzLsrAu55nIx7nzxP7cb+g9R+m9
gL6n6L03XqXzI33x/hTUGfN304X5Ie/sjd83vCIPkm+611nyDbPgenjuQG4M
OCPMD2XlnVSaZ2bnNPKv86bfbZWSp/S7ju4/C+vou+26yK9l5CbR/iH2iVhH
x12k0SX0/zqmJbeY/ONanBBe/yFVsTTfzevi6Pjmpzto28Y+hXT+Klqf9Pek
PcLPP6n9Rlq/z71NNI5v0c/kK79J+42HHXbj/W9ZvPD5MerpNtrfpDOpqoyl
751K7bAs8h3OwbrWD0oQ+kfcprraLBfyGwFjMqjOvsGkulM6rcfc4PKT8PPb
l7+B9vf5DJ9Dm4oo8kf4xZDe7ZlMx3fFLHoXc1TNXI6+9okhlZ8lk2oZXR7R
er5KJ9WOragRXr/SWOprfVQVqRLoSn5l9FLUKXxIasyteSS8fkO7wG+OIDUq
UU9qaUs+03kOqb7TQ+hXs58idzo+Rs60+Bx1QnTkVisb1Ok0TDz+L8V4v3Go
QH9ff4b8sp+O9fgfJVU7VIvzb+QpjD/7PPxfPkBuljtgXQWTSbXSJ8L8k1zy
6Xy5/zHM37EA/spqqNtLzGesRejXXdMx7p40zMMGKtXkoF73Iuj9kUK/8SSe
jhutLbgOV2Mxrl8CxnVPwvyUI8L80z6IhO9uFMa5FQ5fuy3QQKhWliV+f1xV
99w5Hoh5LFmL82+HYn8h6hvPE8Xvj7b+ON8Jqsycj/MHB2A+zkE47h8nnv/V
BVhfOt5PzaMzsf6bi6B2a+ArDxP61UAPjHt2Lq7DizpNW4G6B1Zju2Wk+Pnx
ljGvTKhBXgWR6hnlpLLa7jFte6eTSn3HPBH2z3wdfRXgiDqrHKCGP/KiZQW2
HWsfi/xm8gTkzEsvqEcuNO4OqW77I/m1qglCv1zwArmVNBQ55eMG33ALqXKw
GfnN8r5Cv2J7G/nwYWusY7orxg+fQSp/exR1EtOF+ad4lWD8JpXIvXWt6HzT
grzRvUNRJ7BInF/DTyBXkqFq3FXkXX/kjdKtLrfOxQn9+tAc+DdBDZdsaNhv
0J61qFPdT+hXEnYhp6em47yaDMwjPgv771zE/sWzhH7pRTzO77Ed481KwXzs
U7Hdtm4+2XnC/DRXb4VfgpptE3BeGOqpHnXz258p9OsvQ3H9J4bgeA3qyDss
8MmJqNM0TejXylZgf4o/zg//Fdv+m7B9KBYau03oN/LXoP6nKzH/0xH4/Fbh
eSp1x/UxBm0X+hmGYRrEPyJQi/8=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 87->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 88->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHtQT2kYx39YuzZmx1hyHU6GYdc0MlrC0CHFsJiV2661Dk1abSZjW+RS
v5qy6IKke9vZLlqUVEQWnRKplQx2bWqn06YxS+geLVk9X/uXd6LambU7z2em
+c457/N93vecfs/zvr+f2WrXhY7dDAaD9Pyv9/O/HgaGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYf5jKEty0q2fqxS6806rqkdX/WEtiNP9B/8uuv830up6rXVc
mz/vVltxr0I1mBV3xs8wzMvo8+0vi+pKXf8wj+o+J7u6VY0Ph1eL4jSHCGH9
y7PT9pPvQk0VafQpod/Qv6Gkzf5xKeQk9aHlCVUdqX+taP22Vp++NiOnM/3D
eOWwmtUJP8O8iWhF61BXBUaV6qziZiFdO9Y8oPr/8gbVrTSva01b9SP39rCn
8YtjY1vrRPVyp31f2XRNJ//OI+L6fxXND2g9so9Xh/y620fUn4zbe3fIL0XO
TaLzy7dRHfIb598/yucW5k1F8gih872x3wb6fOsr9mO/H7iFVOk3B3XfdFJY
/9rMPsepPp13VFCdZ7l5Ub3M9M/Fvv8N8g53aLN/vIRFzGDab+08Cyh/dFOH
6k923Uf9Q5IntG/+FygXRwaRf5Ob3pk6lvbYXOuMX3cfac99hPnHiZhFdaEv
S0R9OCc8pH3/kyK61gbLtdQHetyuFZ7zi3pF0PiKa3ReUPImo07DsF8b3VD3
8hceQr/0eFsU3Y8r/o7i8orO0PzvTLmHujOBf3TTa9Wv7LDGpbVvyMssSsmf
Xk/9Tfd/8Fp+xbnUg+YPehpHusjqPj1Hl7DX8qvdLNZQvLnzdpq/pZR+N5GG
dm9X/9HGPHWj+QM9b9L6W1LPt8cvJzQkUP98nOFNvmGWAR36/tTSGMLfe/6/
6JVzqS6l6z6kskUAqR54qxb7uEsd3bceVy/c/3/V6HNtXGOC+j4HlX62gj80
i1TdlVon8iumM3A+bxiNvnHeCf2o1Av9J/t6zYs6EvYPY/rlq9R31g0sJ820
we8N7grW5bkH6hIn9CuF5XS+MHqPov1Zi7KnviM/GYvvPYYNqFsTS6FfmzSC
zjlq7qBLlGdK1zKKy5pKfq3IBuuPChTXv+Uk6nfaRL8s8m/+9AY9/5C+eB/S
UuSpThOfvySrFMpfPy2V1qE0ZVKePumUR+6Lc506urFC+PwmMYcpfsGsQxRf
HkH5ZCcvWo/B4fYvdH3rZoFw/eEj6XujpvvEks7pR3kMQflJ6Jt+Z+n+o5I0
kV9PNAmmuPfraB8x3Bmzh+ar2E335c1Dj5HWtMQJ///jnx2k8UfL40njbY4j
bsYJmneCOZ7L5t2Dwv2rJSeBxrsehD8mmH73VruPJdUmFyfS/a2LjEK/1+K9
tP6gUckUb3Y3jOInumBfOz8B77dn+cY38vy20pHq2rioAPWdUkaqVJs10P38
WFI5ObxR+PnbVUR1LUU+ITWWfYh8zdtI5eGnSfVBOQ1Cf6Up/M4zSHUpABrZ
hHyOa5HH11boV9Nr0V/Wt5AaHaaiz1THkypaM/J83UfolxbeQ12HNKDv+fRF
v9vgQarVl8A/N1vc/3bp8Nn/hjxX/4RauWL+pO+hls+E/U/tVYB+mf0TfA2F
eA6rnngPTnbQBSlCv+HpKfTZAZnQq6eR7/IF9HMfA/xddgr9emgy5psNlc4c
ge9eCq6TcrGusHzx+S05FvPaJOA9aCquk+JxbnyGfIqeJfTr2j6MvxUG3+JI
zJsZjvspyKM/PiE+f674CuN1OxAf64/rZ0G49oyGb1WqeP/wW4L4iUvxORqB
PJrPbvgPYF1qYbx4/bafI84a51spHvuoPiUQ7zHvADQ3Qej/14lyorrWGhOg
Q4pJVUfTJnqubfmk6pGlj4Tvb296A+p1BPmkysnQeC/kWfke+Q3mpkK/tnEx
+kzEKuQZl4F+U1aKerVyR985t6BJ+P5y30b8ADvEm88mVVxPk6rD7iL/7g+E
fv2zGqprza4HxWkHLBFfqUC9fySVLI4L+59cXgH/rGb0KakRfSLUGvka8Vxa
RoWw/yjuV+FPvkIq5ZVAB1Uhn8kUzG/rI+6fJ84izvMMqRp9Cv03/Dp0Ovq5
vNFc3P8aj6M/+6bD3x9q3JqG+6FYn+Q2Xuwfm4j86Qfx3Ga41rdApWFJ0H2a
sH8aPaMQ7x2NfacqBnGmsViH7WGMB5wU+22D8f5q90Onh7/IF4l1FcUhj3Wa
uH9P80VcGVTx2oX34BeI66MRyBf8g9AvHduB8WU+GP8YPoMV1qV64rk030NC
P8MwTLv4Cyk2i7I=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 89->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
5+ga+LJSxftnGIZ5Hf4B9dqH8g==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 90->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmAlQVlUUx79RR/xyXzJzwaeV45JLWW4T+UpzSSlxNCUQn7iAZpIaMwoJ
DxRcymVUUlwfIoJCgCCiIPLQFFA02VxC8SkKSpgk4jKh5nf+TGN1VaRptJnz
m2H+8+53/+fcd3nnvPt97ZzdRk6uaTKZpId/jR7+1TExDMMwDMMwDMMwDMMw
DMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMw
DMMwDMMwDMMwDPN/Q+4U1d8ink4XLKp7jrnUXzTvUO3TwvFKlFl7Yy2fSwkV
6U+a9zQ0LSLn3/gZhvknml1yCtWndd7sR+tLb+pK9arlbygl3dWtVFR/iqO5
QDSuHTroQ32jTdg16iNWIUK/dH+k8cT+kZseTfkrUor/Vf13nLr9efYPxXvg
eO5fzIuGkht+3fJcqtJcz+SHKvWah7r36kbjsncm1a20svZvwvrvlu5K423t
B1G9T2s5mnztIw5aVHkwnc4PckmmsP6firt6gtbXOrBafrnU/QL8Ha5XK3/j
K5G0H1ftqpVfdfMPpX3IW7EuuTr5GeY/REs6XUTPd1YqPd96oR/e96lHcZ07
gupeGhUhrH/9zb5xVF9zh+EckOASSHrc6TA998MnUBzZw0Po/3Md7/V0/Mvn
OaPWU955jY9Y1PDt8kT/49AXrDlG9zF4frX80lJ1vKVujd8f8/2nishn9mTx
+5950ZCP9KO6UI2tpEr+JFLD+RjqZandDarDVkduiJ5fI9m0meprbV+qc6Vh
7K+kheV0rYYOoTj6fjuxvyRjA/WHuvNWkTaxTaJ8A0rpvK8MvUVxjB2Xnli/
ilePD3D+eHclqUvCOdJx165QPEWtUv3rti6R9J4etXkr9cGyfPr+Ynrbu2r9
w7kgxOLXouqtI//u81cpv0+vKvmNAfb2NM/pwEC6/0XdT9L9ZzY//Cz9Qw+b
Yf/ofPXToQu5/zB/R3aaRnWpTHYlleeMQb1nryU1znqVUR0XPSgTPj/jluG5
7q6TasMvoN4LzIizcCepumq30C99GFqA97wV+s+knegjzVFvyvu/4PwRmCvs
H6Y7Tem9Krn7XKT11xlC/UeyikCcesvR36JWCv3SxS4Z9LlenE3zp5+lepcS
++D8U/w1+eVONmJ/fINDNC/Kjc4psm9P+r4hzR6HvnVhBtZhHyc+P325mPqd
er5IpzjXJap3pZYL/AOKcB67mSX+/pU1m34fkZ13xVDeA7USaF4DNZfidXOH
v9TrsjB/9sVw8tvf2UHrqDiP31sm5mNd5sJTpB38M4T7P9s6iPIEnwsmNW9D
nLxy+t6keodRHOXuzF1Cv8+kAPp8RG+8B/JPrsB9N/+eri/H4/5ORoWI/KqD
A32/kr9qQv3aaHSFfm82pVxBvmaf0+8+8ozsbcL9y3GncWWmI/lNDT8mv1G/
iPbTZHWd4ptuzPcV3/+95RTfzvwD+QJG0/lXa1OB++mn0H7ogyZ4vIj9Vxrh
dRN18zOpkhhEKnVsW473djQ05dtbwv1firqWF5RAwwfDP2QAqaoWI/661HKR
XzvxMvn046NJle77SI28duQzbRqHdQUPEvrVBjUw/7P76DOfTMA6Zp5B3P31
kT+opdCvTy9F/1t0A2p+Ff57e9GvrMtItZcO3BT5Db8C8mmvnUV/SLWi+dKD
OYhTIw7+1ZLQL7fPxLrDMqDlUNm1M/aj6Tfov9Z+wv6pDE/Guqfsg2bvhcbE
I07rxlhH4VShX/aNQt7ccOgrYfBv2I77GZEGDU4V9j95cTD6fLMgqMN6zCvD
tWkN4hmJSeLz30eYrx6vfN9kBGI/k6B65xD4HOPE+U3o68qYZYijL8V119W4
jsb61BaxQr8evhh5Dnkgb6Y/5n+xAOvx2wRfVrg4v78n/I284Y9UMe/cEozX
xDq0uhHi99dzRtrlQHWtH06G3ksnVZdJt+n/6XIUGuh0R7h/owyqK3lsV/Jp
OVCjxSpSObYU40bFbWH9RtuW4/8+Fbp5I6mUdopUyfXCemIGCv26bTP4PIeg
vt0RzyQHkGrZd6HjzUK/FnaH6tL47j76VVAnmq/a2cCXsgfxpgQI+5+8shC+
0yXoMz+ZsZ7iVvC1nY/9WXJC2H+kLbmV/Skb6yjJxHUfA32vZnv4eocK/UYO
+pIx60fMfyMVfhtcy1ua4D6s3hLnN/bQPL1VIjQtHr5yjJt6HEE/j/QV989h
O+CzjsD8y5W6IhL7ciwd17Fp4v7XJhDrf30L8k4MgW9WOMbPx+I6Jkns37IQ
47NWkGpdNOzDjm1Y9zvRiKMkCv3KXDfMt1GRP3kZ8jlsxnhEGO7PO06cf7Un
8jT0gc8uAPNbb4T/VCh8Y2OEfoZhmGfiDyDbjYM=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 91->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
eZO+
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 92->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmH1UT3ccx3+e5nG1luNhZ9OlLSc2DnkOXcuJY5x5aPK07TLGzGnzlISz
mxIHiTxP5eZXalSitKHpZss8DCmJEZdfp7KYHoWcTJ835+yPr0ztHOZ8Xuc4
b7977/vz/Xx/9Xnfe2s/5ZvR0xqYTCbp0b83Hv1rYmIYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEY5v/GQ7sol0ei9Ys4X61Sj6NXXETXpf52XHj8MYplS3D1eX3p
gYCarnsmb8el18nPMMwzkb6K9qB5P2V/kua2IKqoWtXLbYtE82c4NM4VHdcu
jnWn403u3SL/gwihX60Xa9SYH8Y0c/V5Jdun4EXMv6x5eXHuMK8qxsfNi+k+
XxniR3N26VACaeQ4zP3DdMxtRZFwfp+gdR+wPaX6P3OWLafcmLGP8kPKvneN
6nufrNH/1P425NF9X7kfWTt/1y60vmnuvlt1mWPD03K7Nn7FPd7M+cG8rBhd
XW5W/37Kw3fRfCkNA0n1iCWYt5YelA/y5phi0e+xPKDzTzTnC49YyDekGb03
qJXJ9F5gtI5HvaZ+Qv8T9ANx21L+2ZdP1Xxa9/qgE6RBjjX6n4bk4nSa8sfm
rVr5nyDPmVyn5w+j2CejLn6tYKQv5wjzXyP9spjmQsoKw3zY52Fee6QV4/7v
WUK54HKyRDj/79YPp/ms9z755PZdoMtCSbWcCVRHHz9O6Jf8ysLo/JTp9Pyh
Lp2STBrmX0h9bWlKfiX3jxrnV9ldRs/pSpN0f/K1tsoh9Yy7Qf0Uz3mu+ZcX
OW6h/g2Dnhv0o+7/yq8lTlpAufdr6GLy3S6n9bXdDZ5rfV3PD6E8dCw/R3XS
Zxyr1fNP96wvq+uo4fu/rlV+DPEfyrnz6qI7KTSXarBGqlsvI9V8SkmljStK
aX4GWpUJ57dDJfKj+RU8J9heRW7MsoP/zV2kRlZIqcivrLhHfz+QO9vB5+6M
ec8JQm78boE2uyzOjwuemeSfWHmd/MMPU+4YU50wb0GBeL/pHi729wo8Rf65
0TRnRrd8eh6SBt/B81D5XPg72Qj96vG2R8nn4ErvO4bhTe8b8jA/vD8dckX/
E/eIn5/WWKVQ/f7WR8jv3DGbfFE3kJ8HX8f3+n2a0G8MXbiXzn84IZH6vZ6E
/LT1oDqmqRX4Pj6vyhOun/xtLB1vZYnBPjruo/1bu+mkX4y+SOedWp4R7t9x
8A7q03ZSJO0zay7V0fuY46mvXpNpf3KjNknC/psXbqR1Jm8IpfNVvvT3Yjmu
HeWvEjCL9qeOWRcl8utz9uwkf5/D9J5leF+i91eTzS36PvRhnXdRvSstooX+
GQHo2+2QGfefKtq/Ka2IVM1dSj5ptVn89+vEzFV0fGYM7Vuqf3Yr+fpmYD93
T9P6ek7Edy9ljq7xwlzfvEgqxZ2Gmh3K6fv87BSpyXneHeHPvySD5lr3RT4Y
wb1J1fneqJNfSKpZpZYLf/6yO/JlxCJS42g46nVrBl/UdPTXd5DQr/RrSNdL
uY2hriNJNWkn6r33gFQNsgjzS75/i+Za9ihGDr5ThbzKmw1/cTrqlmhi/5gC
+NULyLskA58POGEfrgFQRZyf2sN05O14PF8pa6/hc6U11lc7wL/+rDA/tf0p
6Nf+MPrPhuqXzyCv6p9FX0WbhH59eQLWrYCahkLllVDVPxV+twxh/plsd8Mf
G4N13eLgGxKLvq4lwj/4uNg/fTt8KyJx3WrU0yKi8blkL/op0IV+7fBqXLcA
ddR+EdDQH9CXbzzOpx0UP7/eVXH+3npc74z7hDFqB7QC+1LVePH9Y74XfOfX
oU4y6mj9Q+FbYobPN0G8/xeMnjeb5loNSCA1bNeSyqbXKkgXnyBVLT3uCvu3
voq57FRFKvk3Q73c0aT6vCpSzctSIfz5dfuEfFrXIPhTt5Kqk/7E8UsB8Od/
JPYHtsF1ZldSZexIUv3TYNQJLSCVM6yFflNVBXLG0RG+lp2gOVNQL+UY+gre
J8w/+WQ+cm/1A6jzVdSz7Yl1by9Afw6lwvwy7c+Cb0kmqRJQivxsVw7Nmvk4
hyPF+bdBL0PepEJDkkn13uegrV97/L2OEfr1+0no98f9WN9+L/qJT4A/7CaO
u68R+rUTUehz1E5cV2xGzk3AZ9PV46jzQZow/5TMbThusx33i0YhuH7EDtRt
FQtd97M4P/02YV0rqFS4Gde1QF1ldjT20+2geP3C5bguaSXOO67C+m02YF05
HHWXJwr96kD4JC/UUXsG4vOxYKxrh30Yf4n9DMMwz8Xf++eMDw==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 93->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 94->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmAlMFVcUhh9Y96Xaqmkq6tMoLlG0lWgRwWnV2qpxDyCtOlrB2k2txuJW
x2IbAUXABYtFRhQEXJDNClgYxAVFhQoSUUlH5CmWVQQtVbRwftoYcmMVTbrk
fAn5c2fuf+6dyTv/u48ecxdOdWliMBiMtX/ta/9aGBiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiG+Y+hjzX5j6xV470OqXUqR/hdGNmIOkpY4ZrG+BqirjSdeBF1
GIYRkBCx8fH+UhyXn64bqx1dy0k93i4X9Z9y2d8kuq7O7Lei7rpkZV1M+RF/
oUzYv9MG5j+prxXfmqC6+8q2TsJ1GIZpPOrJ+r4e+igkuVa0o5v20DhpAl3X
NxSQSs0Nt4X9X+y5XnRdCp2VSdeLUqi/9daZwvz4O7Svu2RT/7cb3Ci/cfri
y3SOebiy9Lny49qxXzl/mP8bxgQf9FVmEKkxpT/1uVT9kMaa+wKMU6LE/d/0
agTNC99AfS51XuVL54W3bNPpeqEXcqS3q9D/J1LMB5+RL9K0iOok9vekc4Nb
8Xkap95oXP8b7bOoTo8TjfKrSscwei/2j4oa49fb+dH7MJR9kv48+SFnOSzi
/GFeOMP80d+Oe0nVqW6k2p40UnnyDxX0+V9wrkLY/54TA+l6sxHoLw83UmXJ
dmi3eajbea7Yb+cYTPfzwxbT+cNpTjzN++Yi9ZuxIhq55B77xPxoiPFqzzx6
rl6tqY70qt1T+fWxJePq9iF3mjqqTvWXLOjcIG8d/0zr/8WWR7fIb2X1dOt/
avFG8uMXcpOQX4GJz5Qf2hqn1Y/Pl3dudeb8YBqi9FiPvrQKhZbOIJXK95Nq
GR536Pq0mjvCc75NDvLC8jw0OIPUmNSS/EoIckOyDBf6Ve3NGzT/c+SOutAb
WrAP9eIfoN6Wa8L8kGp86PeBoc3aApoXe4vyQnZaSj5ljAfybexsod9YdJ/O
F9pm5xyaH1lVQucVD0vs48Ec+GOPC/tXLrA8Rffn250jvTiPzkHKqHs49yQP
x/7vhIjPT8umaeQzn0f/Z9VWvHOJ5r9/FPm5uQy5Oua62D/EO5p8M+/H0Xpd
Jaon2zpQHcXmMs5xswIKhfufbRZJ85ekHaR5SS4xtL7L1RTk8odXaH8TszNF
fq1y/m6af+FMKD5H76FOenoUjfda0H5Uu19+FPn1rsHbaN/9u9H/eYxNM7bg
81McQPPfHU/Pp4w+HS5cf0MrrOvTag++d4bHUr11Lngf2ql9+B6LEvqlXqW0
f9lmOqkyuJrWM9RMovegWn5PPsntyw3C/DTLx+/fC6P307zDzti36exOqle4
jNbXXnNw/1fmb3ZYJe077mQlnt+b1GBdDb1ysoqu58+8K+wfUxb1tdynGc3X
+3SGrhqOuil5qNtXqxJ+/po6kl/P84W6boWWFJNKp5aQ3zhppNBvMJnTPOO4
vtjHlKHwRS1HnTG3oeE3K4X9M6MKOXXoNvLuYim00Bp1XONJlUshQr+qm2i+
nHsZ56RWt0h1C+xLafYt1k8aJPRr9zOw3oCzqBOcjv14toU/xwH7aG8u9Ett
UpGvbsnwbf8J60/RcH0Fnkd1OiDOX78YrG8WDa2JRE7mxWE8IgF+sxxhfuqp
+7DeqAjMmwQ1LDqEsSPqSV+cEedvUAj2XYHvH61ZGMbDUNewu77OXU3sd9mF
9z55N9Yx7cS87qhrCDyIeocThX6DEgj/zyrmue9AnUEY6zf3QL+LET9/uhfm
5/rhvvcW6Gmcm3V77M9gEyde/x9GKf+K+lp9uYhUa3uJ1FjU/R49V34hqVww
4jfh/heVUV8aB/RGnW5d4J+zHVppRn59COo0RIpwrcL3wxH0t3kSqW7ZHHnz
0SZSpeVkod/gNQg+XwdomEwq+9TnTYcW5Netmwj9Wtjv9Tk1juZLvW3g37qL
VMs9hv20DhXmn7QUuSKVlZHKWa/AX1CfV9Gb4Q9uKfQrzjnIt5B78PsWk6pB
U/BeIqFazDFx/iWmYP7HUM0uDTnheBX7SjOiTskaoV/JiMf606HS2iOol30F
dczxfLJ9kNCv7o2AvxqqR4Qj728ewH7cM3H/9XPC/NLX7UJ9i/p8PbsbdQJQ
TzscC3+T4+L86xmA9ZrvwLxDKvafgDrawCiMTySL83+JF3wlPli3ZBv2YxsE
//X657FNFOd/P/iU7I24X+KP8dJArDsBz6Gsjhf6GYZhnok/AAmqd0g=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 95->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl2tQVVUUx49iPppGg2YMU8bDBx1TwDGd6EPEyUzzTSRmJXgtyUcqiGYo
mscCrVEEZRQQlcMjEHkpIQK+DqKigQKiosM09/hEQLgqgqhgyfrzoQ87U3TG
svWbufOfs8/+r732uXetu4/9lz7u3laSJMkPP68+/HSVGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIb5j2GkfRfi+lC1/Uo26SSp2PU55qNPLd33PNdnmP8Dar+W
Ua11pm9+81irKsvcbpAejLaI6s+wC7woGjfts/VpHVddE6tb1ZTWs040T5tr
vvA4da2eHnCZ659hni2moCyqb2ODzY6DrQO7o2OpXtc507g07TfcP5B/Q1jn
4dYpfx03rPRVdG075BTVv+3SS6T+uUL/P+ZnGXOG+lHT4Pb5q+LOk29k8vWn
6R9aZUEV9x/mRUM7dxb/7+NHk+pZE0lNuZdJ5Vq/m1T/RdtuCn//lWVpNE+Z
S+cAZebhddRHVgYU0bhTNfrHnEVifxuqr9mD1k3/akarXx3UezXFa1mE944P
Oj/S/3coLdNPU/+Zdb1d/UMrG0b9ULKxtK9/uN+f1Lofo8vw40/VP3qa6bkw
zLNEdw+jutLCjpDKnWdDzXeh/bbcovrJK7gl+v2qcy/GkT+nEf2i41H0k+Df
oZnvIc7VkUK/7FUfT+MdCgOp/9hcOUDrzWyhetNdbcivJqY9sv6NrQeCaf7+
sijSlLFmWn/syRryLQ94rP6huJTtbK0zWb0XRv3I7X167zF5uj6eP960kJ7H
bucIml83mNZXfAIfy6+/OyeG1q/JCaLn9s1O9K/pg048Sf8wVR/RqI+OsVpO
vnteDu3pP/Kivkv43PPiIl/aQ3Wp9dpFqqcvRL3XHYXahtbT77DBUi+s/xG3
cD64X3wTdVeOfjK8CnUbtANxD/0s9BvB567SfQveM9T4JTh/xCbA37+UVPE7
KewfercF9H6g5Ay4QvcjVuAcM2wkqTZmBfK64Cr0m2xT6HyhTvYtp302f0b1
LmcOQf/q2XZuWSwL/VJtYgHt31k6Sfd9Heh9R1vsiP0c80acTs3C+jf8snSa
v2vwYZpnSaD3FV3zw3llYxTOT/eKhX55SK9f6XkVxmRR3nJLHs0bU4r3nkjE
0bssEL6/KHcr0inPlWdIlW+bKZ5ydVw+jV9qqaD1z8eUCve/wpH6t9K3MJH2
URKCeDkhGbRuxFban+RiyRb5tROB4TS/t7dG+fdwpGu5OmUrxXW4TnHUGqdk
4frBHyXQvAvXKA/T+rOZNH9LH3oepuH2qZRHoHmH8PdTVUz/X0rFflI1w4T9
r6kg1QvDyaeEvB0qfH4TzvxE6/wYTe/BakkC/f9oqQ60H93LnsYNj4agf2Mf
VT8OvU37W1tEqi/Ix/V8qwbUr5lUy/qiUVj/Hhfr8X0PJZ/W/3XohB9uo37N
0LcONQif32hv8ht22+vxPaah30yyRhw/X1JTN0eh34izovn6oimIk7Ic16NK
oP4PkN/sktvC79+mEXW9/w6p6Q1HzI9PRB4uZuRVNFbolz2qyCcfvwC/Zwup
ERmAOKEp6HsvDRT6pYXl8NeVok9ea4Z2nQXfVV/k4VYr7J+6k4713KFqQD7y
yL6Bvh5mh/xdtwv9yswsrNd9J/zuUFlqGx9QQarMKxf33ynpWCc5CfNPod/L
vmkY756JfP44IfRrY+Mx3yUO+/Bq04hY+K6k4DojX+g37MIxT8I5VTq3CfkO
jML6zshLO79X6Ff6bcB4VCjyPxGG59ADKo1GfnpQhnj/Q1cj/ieIo2a1qRSJ
fYViH2q/HPH/x/Mmdk0j+r6F1OSXSqot6HuH8v7wIjRyYJOw/gtrqC6NXCfy
qYX28C9eRipbv0J++bXqO8L9R80mv967AP0mdy+pqb4JOmsTxdG/Hyf0K7XO
6E+9PoXWzEc+47JJleU22JdtN6Hf8GxBf2l8B+vnuJDKrr6IM6Ic+YVtEPY/
o66K/MbhJvTNpT2QR8EI9Cu3aOiqTkK/LpWj706shn/2ffSJ9XbII9kfeQ3K
E/Y/0+pDyN/3CPwd0b/VtWehU3vB9/kqoV89m431+uQiTsAe5DO5GPsqrUec
r5PE/rRU3LekYP5SqD4jHfHWFCL+g5PC/qdYJ9K4nARVh0KNeW1xknbD53JU
6Df8f8G+F7ZphwT4X0YcbXwGxvPyxP3XcwvmxW7DfZ9o+KfF4DoY+5Ps9onX
37sZeeqbsI/KKMyrRBy1KBnjG8V+hmGYJ+JPChOQEw==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 96->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl3tMVmUcx1+8hZcCLSGV6VELM8NM0/ACnAw1UNO8tGQQR5z+IRguLxuC
9i4DyrQpeIfwgCgXUZHXqYjIAUE0NQy8T+CgECIDFOUiqSS/L221PZqim+Z+
n41995zzfJ/n956X3/c8b28v3ymzWxsMBunBn+WDP3MDwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzD/c+Q8z0yn510EwzDPlq0/L3aCBv2zv9U346jflYbZN5pU
nZteJez/tmb6o3JBqY0ta7ovTZxbKZqnpG0qepxcUbtZFXP+MMyzRTN1pv42
JJglpj0Q9Wi3kKaxHu2Dvl91Hzpj4w1R/0l295Pp/h/rljepZjE/kDR/el6T
GrdsvEr3xxwT+v8LaciIM00+ObFDi/xatvsF2v+VgutPkx/qyhPXOH+Ylw19
VCb1lRTag1TJ/B26r89N6rthY0gN95bfFL6/j+bsoT7/Zsm/3uP6KfUUjXOK
aT15hb3Q/9C62g2eTzlSq+dQHnl3eiL/36jKiLPkD+vbIr82OVKl/Pjo+4qn
6X/9eEw25wfzoqGtPEx9oVyaRqrVjCSVbBJwvXRtNY2D9lYL3/9tLeOp/0PM
ab6xzVnqd215NXTmTOTIgbFCv9yh41byVQwKoD7Lnn2I9ou2p36T/rQivz4n
+5H9q5V2xvlj6PqVtF9acgGNC+vL6XNE+Leo/xWzpfS7R7X+7vH8v9h92XSO
kqZ7+9H+7dfQuUNpPe2J9jfedphMvmRXOkfps5adakn9+irj2KfKLde+3pxb
Ly/GVRfQl73R33LaHlJlcKtbpFN3Qy3Nbgv/D7Qu1ej7EvRpzHVSNcgB11Oj
SDXniFsiv3opoJTuW7uhP4Jc4b8cjvVSX0M9cUnC/FAbJp6j+a56CeWFZQrl
ju7ijNxZsAK55Ogl9BuPDj1N10uDzlMdjfOo35VI/O5ReuD8o8uthX7tx0HH
aJ8yezqnyKY7V6iePuOQp37DoY0P8df0SqfrfW8foXn3zS+R+iBHjSkD4R+W
I8wPXQk3UX0LAvfT/ZIkDbk5gdZRHCKbz3cTxL9/Ojol0vWkz0nlquC9pJ80
ZNC65kWXaX/T17nC/F6xKZrmu7wbSzryzm7ad8bNJKrftSc+3/T8ZOH+idUb
aP27oThnOfnRWF7sHEHzB42hz6dVOycI/bvmbCffyVbbaN6sK6h/8Dv7aDwq
dSep24Ydwu9/dSjeP7n7SQ093fA871WSykVp9H6TSwLWCM+/Myp/oPtvTaL6
dD+bMFrPuWwLjYPd6bq61iP4RcxRvSGV+tp4NZ9Uib0ADbCqoec2JJdUmje6
Vvj+755PfW30sCGflt4dOfGpO6k0voRUfz2zRvj8+02BX/Mh1a9vJpVtC5A7
v07DesschX4tugPNUycNgNo6IGeSA6FZVaRS0nlxfjnUIfc61qM/l76NOrxs
SbXyg6jn2xVCvxx1Dfm2pZBUX32GVB3XgNwa6Iw6bPoI/Xr5OfijcuHzLUY9
H9fCP787nk+hQehXdmdhfrtMzD+SgfNap6vIX3+ooU2WMH8N6j7s23gAPt/m
8bWD8Jvn4HOF6eL8DWl+XwxJxPxOUMkN6xiWJcNvyhWf//Jj4NuzA/pZAtaT
dmN+pQlaf0ycn++HY7/L0fB7RuJ5VMRhvg3q0ftliOu3xflW9tyM+4kRWC9j
G3x+8bi+PkXoV5b+hPljN2H+uDBol634XjNQh9L7sND/vDEuCqG+Vi8Wob9r
4kk1//Z1pP0LSaUB/euFz9+7HvmQ/CH5FPeuWC80gNT4W2fyG70u1Amfv4sP
/AV56O/0dKj1XeiGMKzXfrzQr7nbwx+gkKpZ80iNATm4HmyFuhyLhfkll91F
X62bSvOVpNHY13ERqR56EWrxhdCvmMqRe69aYF9jV6yzcAHGJ3dh7GEt3t/l
IvavqMQ6UxpJ5RN28MctQT15x4X5J7lnIl9NUMkiHfltVwVdMoB8squn0C8H
psA3pFmr9qMej7Ooo1cddGas2D98F/Y/lAD/iTiM++/E/huPY+x7Wphf0hux
uP5eDPZZGA0twXXlKxPul2SJ8/OD7XRddYzCe+y8Cv+tSMyfhjoMR9PE/tgI
1N0WaryyqbmeMFyfhTpU/0NCv9qAeWrYetyv34jv0Soc/qh4rNtT7GcYhnki
/gIh2WU6
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 97->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
8OvV2x+4rwzDMI/Eb7nlcHY=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 98->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmAlQVVUYx6/R6HMrzSVzyStpqYVLYI7jwnVymULNKEWN9IpCYkpjRIFg
3lxafNmAQijaeJ8CKoIoiGKiXhA0lXFhUzTlqiUoCKgYOI6U7/tbY85xbybH
+X4zb/5zlv93vnN457v30cHrE3dvB0mS5BufJjc+FolhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhmMccOX7C9J03VB1k2+Zqb2eUldlVj06rcBXN91lxVNSv
Bn4YSHHc0n63j6vpX54XzdPONTdF/f+Mt8p+xz5uzm1013n3wrT6Dn4UP8M8
iZgZIZV0P4/FbrHfV3Oq7za7ytNU6jd8bKRyF/9K4f2J65ZF/dYj2s5bujVX
F6oL6pw/f6Pxp9LE/nugOSl5lIfS/6H8SrvkfMo/q0Xxo9x/Rc88y/WDeeLY
eAz3/PwUUi33s4v0vLXkoj+qO7Vla8hF0fdfr34W7wlhjqdo3NRXUHvq6zmk
zXdTHLP9K0L/32gOsf7C8QZtD1H/OYe7+u+EsSOrgPbl3Oah/NLZBmHkb9xQ
+P5zL3Rrfm97XTTG19v7KPVDf752Ltcf5r9GuRiEe5GrkWpnvUhVyxJSo9fX
l6jdE3o7WpF3sr1fmTUWcVpZoKvqom5U+iJeQWehX+7Wai2tG2ANpvX8P6Z6
os8eUE7tJYgj/Rx9X/fXdPtmDOUT7k6/FxTvVRfIVxh3V7/xavI4ytd/FtUh
tch5KbV9gvHeUc/zvtaX52xef+t7kNoklH73mN7y/eXf+OjEW+fJHaJyyV+w
7dCD3H/zcuOAf81vui+C6wdzO3LVPrqXcssIUuPaAKh1J+5974WX6T7qlZeF
35/uuagTB4rx/Q4rgobXop6s2UBxNEuQ0C/X31FC688NpPl673WkSpsk6M5q
vI+4nRTWD7WiJ/7/UPUdvd/LV5siTteZyCP1e9SfYh+hXyoso3tlLJhYSHla
J9F912PK8f6TebN++TcT+o1F/ei5rlZWUBz5UiD93pHzB6P+jfbAOdTmiN+f
HMrTad6Q8fQ7SrcN/JXaR2+uP6Ij4nR8X+hXslduIt/p+ltJHWdn0Px5lcdp
3eJkxDmdVyrc//XrG+ic+w/dSOecnZRCvhmtM6k9PO8k6ejyXKE/NC+G9h+9
iOq42ScX8RZ2pryUfI32p/3QLE2Yv9QwivIbdcqG50QvaqsRpk79C4dSHMOh
dL3w/CzTVtO4bVgsjb/2BeVveOzfTOcwakQi5XVkWoLIbxZ4RtP8mXGk8vYc
ep4Z1wNoXemjHvG0j9r4cOH3r2aMlcZz8mie9ILTcvo7fBVJ+UurJ9C6Smyk
9XGsv2pJeRXl5Z5Oqnqkof1S4yukiZWkes9hfwif/8OeovlG+WT4Go0h1Q+G
oT3/GsZPnbwi3H/SZKoLanoq6szEbFLtGOJK24MR70xfoV+e/gx8rQNIzZaI
J9eLRNzDp6AhZ6uEf//2tXSv9RIJ9SmpI+IN7AWdtBL9tv5Cv1FYivqZBTXm
XyNVQl3gT5iD/aR3EfrlhcdovnnhCOqLewn0eAfk7TQF+wrfJayfZqdfMD90
N+L0ykI+3YtQv7e0QZxAL6Ff67MN9Tk1BfMPJEMXpCFuS+Sn+92h/vpgvvFt
AtaNise8BoineKeif9NBoV85uhb9KWuwTmkM9rE4Dv5yPD+UI3vE9fftVcg3
Hir5RSGOlw3tRORlDskQ+vXCnzDeDz5jfyTW+xH90gfIT0/ZLn7/dQ5Hv7YE
efgthZpY30zHPvROhvj58z+j+W2he62EF0OvFZGqqa7VdB5vnYG261EjPL+t
NXQvlQl94R85mNRwDIDGXUfdSK2pFvrnesL/RjKp6bUM9ebTC6Sa6Ut+84yb
0C8Nf5PmyZNGkhqD/KEFixHXchXjLbYL65dqrYPxhl0xf5Uz9MQs1JvMDMQ7
EyT0G04VqE8z2mDeyfbwu3SG/4Qf+tc4Cv261wnyKyFXoB6IZ2xFPLngXZxH
8AFh/dMS9mL9tvtRv52ySc06JqncqSHW9wwW19/nMrBuNVS27ECc9/JJtbXF
qFuW9UK/WZSM9Z9OgW8Z2loWVH3xMPzLc4T1T226DvOXrcf6Lydg3rwk9I/b
ivNIzBb6pcnR6PeIgf/8WrRdEFcq24zzqLtH6FfGRmF83wroYhvWi4yFBmEf
5ue7xPV/+jLs88pSaI2O+RHwS1HYhzEnQ5w/wzDMg/AX415xPg==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 99->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl31UT3ccx39HY1rm4EwkD3fHw+yUhuZoObjDMGdDrAxt5yr9QRoZC6vt
etoDRR4yT3FL/XqQip6IcnsSEifLQ566GylaSKU0junzdvbPvrOwc+bsfF7n
dN7nfn/f9+f7/d5+n/fv3jc95k7ysjCZTNLjv3aP/1qbGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZ5ydBuWUweDvVsUqlniw2HH6u+f20aXcc13GxS+Xxw
1XCB35BcfxaNyx8PTGqqo1mbrjV9riY2lovmScf8SkXjf6lXE3uxOfMYhmk+
+orRd6j/C64ebOpXOf9qFvW/m0LjakY6qbJ5+R1R/6l21ccoH2zTVpO/cdBq
8rUae4G079jrlCOnM4X+f8LYG3CG/LrHc/m1jrvg929R9iL5odrXv5CfYV5G
5Eb0t1pYQSoP2koqBY6spr6fNQIa41Et+v5rFnU6fV5da5BfUyKonlPjacqV
E6OonjHYWuj/k9KT02he+QI/8tnM+Ynq+U0vovGMDk/3/w3qdbvz5Kvp9Ex+
dZYyj9Z3tg1sUt3b4rnyRyk6GdSUi2plfP4L5U++nR/nD/Nvo/kcor7Qh+xB
n/sn4bprPfrFeeZdyoOl390Vff+UzPGp1Cehb9B8KUDD80JgAKneYzbq9ekl
9j/6ZA/lxeudN9K89A6ZeA9JvkXzzZvxfLI75Kn9K/cOWEOfd1u6vKnfjIqN
lEf6u5ZURzkb+lS/FHJgBvWp5+1R5Otv2kHabTLWP+rUrPxQPxuVSM9Bq5wp
v0wJEyrJ/+bCZvn1vIlh9N40Uw6m+Yljiul+jIwuepb+1w9V7qLz2Jb6k39b
4PrnyQ85Y5M7587/F8U3m/pSzUlBn89Fn6uWCaSGfVgNjVuV1oi+B7rHDXyv
XfrRfHlIS/S55XhSvV0JqfbDEqFf/nXxDfp8TBWeExy2Uz31izxSeWMn7Mvl
kDA/TFu9S6i/U30r6POWBuq8NY/82rdbkD8zpgj9Wnt3PF90NtP7itK2Gr/z
75fjuajAB/vJvifsX/3BTXr/kWrjqY4U/CO9J0jR4XiOyj6C56rtOUK/lDs9
m/a5z3wE6w+7Qvt5kAP/pSj4Z50Xr3/y7RQa1y8eoPnvmXNpvwdSLlO9kjPI
r7BNvwnvn8f0veTrcYpU3zkJeZ7olkf+2OWluJ8FxSK/mjzeTP4Sm1ia12MA
1dF8o5NpfGFPOp/acVCG8P5v8d9K4/724bRe+mJc+w4NI98pjeqY+lsmCu9f
UnAU7bd+Mu3DlOuO++GTRucw1YUkUJ1V3eOF698NoudVvSgcz63TUpJo34u8
aV3Za1kcnWtyRYhw/do19L4rt8mgeaZ1Fdvo2nGlRjoggn7f9LR+QS9jjioN
F2tpf+4FpOqcaFL5XKc6+n/czyY1dXe+J+zfBfXU15qNDfmU03bwX3CB7r6F
+o5JdcLv72Iv+F9dSapk7UPezK0hNSIV7CvKXug3tWpD8/QE1DEKF+G6TTqp
PKEB9SaeqBWuLz1CLkTUIq86dkNO7Z0OXXYUGjZN6FdWViPfHl1BXvbFtfLl
BJzn8mqs7zxU7P/9MtZ3OIP148qRd5v6kE8tnIg65vZif9VxrF+Xj/xtdwTq
UIUcn2qF+5vrI8xfxQO5qrsdhFbuh69jOs7x+Smca8tZ8fObVyp8pUmY1ycN
GoA6hiPUlFEszm9zPNb5BirZ78N5cqG6jnrK4eNCv3x2J8ZfiYHvq0jMj9qN
+9AtGXVX5Yn3P2wbxseHYt7UCOzbFfXUIdiH8kGW+PdjxwaMB8Gvx4Zh/pP9
GIX4HdUO5YrP/x9jfHiW+lqyrSQ1MsrQ5+Hj6ukcNjWkRoNTgzD/uram+Yrr
VFJ90mxcV22Gft+S/Pp6K6FfGb0QOXP7EqkUnI/roQ9I5f5BVEfr7lov7N/u
42ie5u0CX9USUnXTeuRFjhX51fW/iPNroAXWjfsI8we8g2u3r1F3XQz2MWWG
0G96rQ59WdaH5ukjkFPyp6gjn9YwvqS3eP0LpcjHXtXIuV4tsL6jFXxbPaGD
I4X5pz18ktvzC0m1Fsg54+EV5K8l9qVl+gn9Rmo21h+dBd+BTOT4/FPQcY2o
7+Yqzu++KTh/ZBLqOCSSSm0xbniew3hGsTi/VsShfh5UGxCL6/vx2P/1/aiz
+bjQL7eNwudlUMVsRp3iGMzvin0o0flCv3EtFOt02YU6C3Zi3qII+KxxLskx
T+iXuuzAuNM27LsYftk6EvXW7kOdOblCP8MwzDPxB3zlhR4=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 100->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHlQVWUYxq8pjZqKuF0U0WNq7qnVlGjKEVxJJ2DGpXHpIkMuOeCY4wSa
HMOF0HDBJRPwIIobiAGikskZEIRERBZ1BPW4gpbsBIhk3vfRpma+VPSPrHl/
M8wz59z3+b73u9zvOd+93WZ5uXo0NhgM0qO/1o/+mhoYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhnlF0fovcEn8y7WcOXG3vVnnq4VmlVZE/WIv8KlvDzwn
uv8nceo18+tazdFbojplleflp/qf1H1vvPg8dQzDPD+KTVCxaF/pobF0XzOc
LDWrsnNZqajO1C31zF/v64nDl1N91NFLpDOa3DarqXO80P9MvMfnmX2qf4cX
8puia3PNPr3s7PWXyQ9lcHthfjHMfxktUC6hz/UgT9pf2sR+pHrukBJcjyqj
/Zc6o0y8/4OTqS5rnE56NXct1cnDaN8qxdHIj769hP4nKO7Wi83nD0Wxm0/n
jrlum6m+b7tsyg/7IU/1/+O4s47SuUHJbd0gvxzh42zuR5s1yMOskp/ji+XX
k/G8C1NfKj+S/LsmPruKYRqEMsm6nPZ7QgjtD7lyCanU4hapkudMr8sZ/uWi
z6/mvvEY1ZdaUr1e1Qf+2Hrs+zUf4v6QVkK/fOZAJNUP8NpG+3zcokSqn3Gf
zh9qbBv08cH2p+5f6XZE4N/OITkt6XuH3twf55icLQ3a/1pZDeWP1Oo0rcPU
yaph+TGs/wbyd/LB96bVrg3yKxO6rqf6rnU59P7Hjs9+kfyQf7/yFb1/Y+o2
vIhfta+cxuee/y+mlGYV9P+dfI/2pzLiIKm6+DqpVBBCr2uL8yqEn4MUK/gG
NCXV0rLpc67tqUAOBMchP0YtFfrVFca79LrRAb7R7bHfB8bi3BF0gdSkRArz
Q704j75naDH975Bv+EXar1KvuzjPOAUg1+58LPQ/+qJCv19ItVvzSRO3kU/u
lYB9/3Aw9m1tD7F/odfPVNdyDo2j6u/S9wTZfiPOCwkPkIPGB8L9r8adTqL6
4x50PpDCt9LvIXJfP8z/Zj76sb0g9OvTjh2mnDPuSqD51c/oPKYXPcQ4pWvI
r9Y43xPm97qsH+j1j2bGkEY1OkJ1BTYp1M/qnjjXNT6QJ/S7DdpDfbrZHqD6
9xJpPMOmJdSX5B5B61POfXJC2P87U4Kpz4GO4TTP3Rt0rTRbu5PuO4zC+ppl
HhK+/+Xhe2ke/UvqQ02/S/WG0Q60Dm2BC3zpQ6KFfotQ+p1L89xBqq4bGUfr
ySkklZ1DoqifuoKtwv4X1nxLdRWLqM4w2zuE1MU5jMbd6HWQxjPWrnslc9RS
r6S+llwk1bOvQ61sq2jdE8pJtW3DfxP1r4xuS/XKDmdo2Tyo3Q5SKcOAcSz2
Vwmf/8FulAv6yJWksncacmKuDfm1h76k6uW2Qr/0XTuqV5cNhb9kHamUmY5x
rlqiH//iSuH8J1+Db6c1qWnKdFy7LCdV9p3FOOGeQr/JtgL5V1eJnLMehHF6
OyI3r30Kf/5MoV/drpPP5HMF+VJZhXH8umAdTcagj7AiYX5KYzKRu0tPo485
GdBJJTjXZVhgPeNTxPld/xP8VT9Co6FyQiLy3+Umxiu6Isw/KfAwnhc5MfA5
x0Inx2N+9xPwTc8S53fc41zPhsoro6BH8BxSfDG+3O2M+PyYFIp5Pt+N/ueH
Q7vvQ/+d0ZeSkSo+vwZsQf1sqClIRf1bEfCPPQRfwElx/9uCsP6iYKh/GMaL
3Attj/kNPU6Jnx//MpKxkPa19P55UiXvOK5z7KpJnS5V4/PXsUbYf1oT+Bo5
keqHXUnVob4Yx68b+bXYe9XC/AjzQb68XkAqdzxLqi9rirxZv5ZUtnIV+k12
TvDdnAoNCoDOjkDujLAlv5b8qzC/TIEWmD/tC/h62kEXepCqcjz6cZwk9u+q
pn0tZ7SgOtP9fpg3aSyuz8MvJ1sJ/VIh8tZgLIGeqkdeTbVBX1/PRO51Eeen
ySoT9ecy4J+dhdx1uAD1GYp+9n8j9nukoG5oMnRZIvK2TTbW1aIK1xMsxf7R
8ajrAFVOxKCPcXG4Ts/F8yTtrDg/N0fSfVNxFOo34VoLi4bGH8X9+gyhX/Pd
g/EtIqBd96EfGao2R1/qG6eEfuVGKO4vCEddn8fPreTduJ6M9SirUoV+PR9+
1ToY8/dWMb8Rfrk73getudjPMAzTIP4AQRZgGA==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 101->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHtQVVUUxo8a5YCYilhR1lH+iMgRXzP5GONoPkhCk1TQm3agSJIxX5D5
ipOMklkqBoYyygUSNANBuTw1jgICYvlAyAfoUZHRBBEFITVN1pdNf+xUcJoe
s34zzJq99/rWXufI+u6VHr6zPP3aSZIk3/3pdPenvcQwDMMwDMMwDMMwDMMw
DMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMw
DMMwDMMwDMMwDMMwzH8EefKqda53o25acb45mkdrF10FefqK6CLR/h91Ai+e
aj6XU5aeE+Up8d4V99PfwzywY+nD5DEM8/CYU6yrm+dKW7XZK+dP+0ZCeg3N
7bgzVyjGhFwRzZ8xb8gB4VzeuXyieV/Rs6qofly+UP8gjIiPaO4Nf9tW6aXY
kUep/2mLjFbd72sf2KxTez1b+Sj+Y/514Rz2L+bfhlw7g+bKeLIfYpMzRbVp
Wi3NTezoOvq9vRZYJ/r9NRcf2EdzHhBA8yUP2zOXfKRdaAnth+7A3P5oL9T/
FcaYoM8o3xR8hPoptG6R/h5qnhX5kO5Z2iL/kG/kRzU/hzF4mx9Fi1Pr/Ke6
fwS9F1N04aPMvzzg5bCcB6cxTIvQZi6+Sp/PT6TTfJmj8ihqY44jHh2O89fD
rgp/f8NcdtOcv1JH82GOnEA6pcyOonFhGeKHzwn1ivcH26h+zaS1pA/trVPc
tJT8Rwm5TXW1uu/vO/+6d2YwzdkWD/q8liaEn6F7HY3LFHdFPJR/yJl9Z9J9
2bFLqY9+4+GLF99skf/It5JCyXd8Mn+meqbklvnH1zPoedR2b5OPquPfKGmJ
XmsI8kd+SsWrzfrZ0WGt8R9l76ZY9p3/L+au12kulZU3MeeTMOdaRSNFuSbk
Gs3B4bPXhPMbKlOe6jAO+UF9Ua/QhDqfJ2LtuFisf6v9JTo3jYL/HHCnqM+P
Rsy3RZ3HEoX+oTutPEl5bjLNmRRZBr/w7Aj9khjM7dC5Qr1q50jfL3T33HLq
f8EUzGmgFfQFUYiLRojvD+5dTP3P7kF1FBud/l4iReD/TYplOunVM1eE/qFO
G5VL9auCCuh8tj/9vUTfr8N3rrpBP6+N8H7Z7JlG567F2fT+znvlUd4nbU/T
/Wv34nlu9rgsvN9h0A56X9OlnVRn6r500lWvoO915jY25KPK0toyoX8cPZRA
eu9i8nHl5ACqJ8XbU1+6aww9nxK+MEf4/t6bvJHOu38XR+8/bTutzbu6xuHz
Z4eF+nrXJ0V4f223rZQ3q5D6ULdconztY1c8x/DlydRHDylZ+P4GW8fTflD5
Zopuo1NJ90VDKt77i0mkv+22XqTX3CtXUZ7H2kT694qzUP/66KAYqrO9lPRa
YeOaR/n+93chu0xsoPftdaieYmY9Rb1zb9qXNvxAUS8Zf104vznPU77h4ENR
CzchSgEUlbDHSa+tsTSI9GYXX/IFedcORPkkRe2d2/CLoRPR162XhHp1dRfK
U+tM0I/UKCoOm+BbzlWoM7CpXqj3b0vn+lM20PdxxbrDBER1K/RfBQj15uEN
8L1rVZjPPR1Rx34InsNpNfpz8xbfH34O+rgy+GivExSN6y+g/6Pvo15WvtA/
pVMHoZtVDJ+cmEtRn4p9aaUt+ji+TKhX/XTk2+xGnaIM6DpkoQ/3Q/Dv8HLx
9z/PdJxPSsP5EkTVDft6/2z0NeCI2L9uJiFvynbUSUpGvnMq7h+HPuRhRWL/
dtoC3WJ8PqgxWMvnUEdKRz9ywn6xPjIG54viUOd0PPKOoZ653oL93fuEemNI
JM5Xm5F/HHXk7G14Dh3PocYXit/fP4zS/jDm+uk8itovRRSNE46NtC4rpWhU
2jQJ398IK8pXXcZS1Lt7YW0dhXWUM+mVhpJG8ftbAn9YkIj5rjwLP4q2QV+d
41Bnfj+hXq72gL7LTMThy1CnqAR18jpBP0cS6hWvbsj/dDpFNdqHohK4nKL8
02Gsa+aI/S+0ET6XYEd5xlj4lLHGhHpZX2LtYivW96qE31puoY7PDYpq4Gu4
v88oxJ6XhP6n2cG3jfkH6zEnx1AvtxzrjD643ypY7L/PFMBfA/Kgm6yj3jro
pQwD/cT5CvWSXxp0zhmoY7UT+ZsseB6PCsSmI2L/TP8W53oSzi1YaxewVqyy
sP94kVAvh8Si35xvcP+dzdBv3Iq+zv/eR65YLw2MxP6gaPSdsBH6AtQzlqdi
LRcI9VpgFO5J24D89aij9YzHeiDuN3LFeoZhmBbxG2DIZGs=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 102->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmA1MVlUYx18DRdGZVlviB160GU5jpiAoKdcSLZphbmVq6B1iqEiSDUPF
vIpgpWmpSYjhBfkQQr4UEDW5iCiimYKioeSFgeY38vpFCibPH1trJ0Jdy7bn
t7Fn59zn/5xz7r3nf8+LnffscdMsTCaTdP+v0/2/tiaGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYf4nyEtd57ndj4acX9kY9TUh59xEeRPD80T9DzA0j1Ok
d59YIcrTe0Sfbk7/R52ZzsUtyWMYpuXoByMvivaVGrrsMu1bl+M1jVE+sbZG
uP8XRB0W9SsjxtO+V906k2+o3kVC/T+h6U7HSZ9QefVR9LLlgpJGnWQ96dSj
6FWLLDcaPzO58nH8x3h/+Mjcx9AzzL+BYfanfalEdaOoFv6AffaOcw2+287X
GqP2VNg14f6v+PIg6VU7g3yiyPLjxvdcO1t3jPaddyDV0b7tItQ/QLPq+inl
D2u9lPJKnUZT26+Wvvva+TbN6v8ORXq7jNbRO+WR/OcBkkXpI+nVuRUhtI6J
3oV8fmGeNPQbHWrpvZyWT/tL7x5JUbqUjX0/bwxd1+dE1oreX83Xic7/ulsU
7Q9j1Trsk6J2pJdnB1A09OFCvZL4cirlmd7YSHXsf6F6Sowt6ji0J70yJbzZ
/S8N0H0ob7LtBzRv2+foe60O8SM/k+clNK/vEz+F1v1sWUCjf8m2FwJI7wHf
UEK7t8h/lNRFS5A/lnxM2zuCzldSvVOL9OrclcE037q45Th3bMD5pXpFycP4
h+pYpv45X+m9Moz9h/krmsUd2peye3czva9R+WintKK20nEFRdn6oln0/hgT
mvyjYAZFZYYrRXWwF0Wt1xX4R0KIUK8OSKP9oThNhV+8NQv+Y8Z+1RoaELve
FfqHKTeC/n+glhy9QPq4KvxecW8Fnd0q+EfWXKFe6hZaDH/JLaf8cz4498Th
d49xdTD2rYOrUC8vyaPzjzp6PdUxwnuepbbdi9AHDoLesUa4/3X/HflYb2s6
H6ie9mcoujjgPHbEBvMP6is+fzXUZtF65cW76PqZrwqo3oQdqJOVivXYn7wi
1PffmQG/XL6Vrt+bvx333XEf1V09iP5vIxmjTgif35YtmymvvCSZrlfso3py
pEHzkn61o/Wp5ybowvGdd0dR3roNsbTOBTH0HZDbuFDblPEm1re4c4bw+a9Z
k0jrk/1pHqqFO+Ub1RuzqX+6OY30Q9U04fP3nhRP/fHuFI3ylduoPdkjk/RW
1vR90oc4RwqfX2DlKho3/tIWul5p0HqM2B4xdB+DV+P79orX6ifRf2X99Rs0
/6TK63T/LdMo6r+9Sv2mBA3Xb7ndFK7fsxflm75XKMpl3oiWX1DUvrmM6+Gb
bgj9wyoYvjMsHT4z7hBi3w6kM7xXUlQDbYR63dyF8iWXydCVh1E02pWi3Qd1
pKG3rgv1shXyn2mP/NMuqDfgM7TN0GurgoR6yXyTfMEIroU/aF1Jp763BH53
Zw/qpPgI9XJsFfTnDcSsOvjvh56o45cEH376rtA/pW7F8N17hxCrEOV4zEue
6oD7ezlc7L+D8uDT7+6ATyfnQJe5G+1ZmJesnRb6n1G9HeOWZaDOkXSsIwT9
Wtgu1LE4JtSr/mm4vjml6buBKAelIr8+G/2hRWL/HpyI/AMJGG9dLPIamvpP
bUP9wALx+XVkHPIaopDnEoP1lKPfKMC6FKv94vNrnw3QWWrIr4/GfGdgfM23
6b7EFYq/X/8xyk+XaV9rmUUU1Zw2t+h+7OuHOP4kRaOn7W3h+qdbk87YOxN6
Dy/Um78W8aUXSK+Flt4S3v9lC2hfa8XpiIcuUVT6mUgvWy+iKF1yEOp11zHI
L/WlaOQuRXtzDkVpixXml1Iv9C9tYH/K0zuOhr/0D0VbCkPb4QTqHQ4Q6pUT
dfDLiLaYfxtbjHt7GHT7F6G9wl7snwur4E8R1+F3Y57G+GN7IVqNQp2Qq0L/
U7yOwp9smuLCw4jPl6NubDfSyY4fCfVa5X74tR+ifmQPYmwx4mtN34XBSWL9
J9nwZ5vtqJOdidg6C/MI/RnXk4+L/TMnGfN0TIfubArG7bQV+s93ob31R6Fe
C4qF7nY88qsTMd85qKP034nxvy4U6tX49eiXNqFORDSeg5YE3cgcjJ93UKg3
eX6HcXwRZR/UMQZCr6lN9yX6gFjPMAzzMPwOVtZnow==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 103->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 104->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl31MVWUcx69ogZBJvixyoqcwXZFThkBOy+OQMif4ukap84hpaqSYxFRE
z3yBnJooAktRj/gGYYiBvPjGMdGLXUV8QVHEjqKiIAmogKgr7+972/zj0RTb
svb7bHffPec+39/znHPv73uf+2bQtOETmptMJunhy/nhy8HEMAzDMAzDMAzD
MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzD
MAzDMAzDMAzDMAzDMAzDMMx/DM3N6Xy/hyqnnivr1wS/ImknrT6lXeA5kd+I
2lX6VHUP+Vqasj7DMI/HONj3mrWv1CXuaq71wuivQq2qVAbeoH5znFtNeiGy
Wth/cz489uh1aanndOtYT0wrsao23yin/IgpEPv/BrXDy6fJ1730ZlP88v2w
fNrH4QFFTfHreYPGWZ+HdmV3yfPkj1bVdSLnF/PCsSAefemuQMf3IlUmda+x
qhHgTyqVKzXC729hygnKj4SBv1GftUmJofx463fqWynKn+pJv3YU+22op/2m
PPq+dtctjvYRVEF9a7R0fqL/cRhKCZ0v9KDMZ8of1X+yYr0PaUSHWMqvkMtN
yq+/kCoH6s/lr/JaxfnB/NPIlkjqK602ilTuPwN9Vp6Dvu88opa097xa0fdP
cWpNv6+me17wXV+Gek4x6Pvt0TRWR/kI/ZLl43S6Hv3SjzQ/Q8+jvm9lkF92
TYDGLX5i/+vdJlBeqAfs1pF/5yv0f0W1D8Y+Rs9/cn74rU229ruxMnQlzb85
LJb637ua/FqLlk+XPzMrulD+nUiOp3PDg45VlF9ZPk/l13pnTch9ZKzE1Z1E
/nk82/klLDqEnlts8CJSr2+nNen8Zc5ROXf+v2gjdtWiX/PQ571mk6o9j6Jf
359zi66HpN4SfQ90kwvmNUZCp/dDnfgwaLaJfGqnELE/9D30h9NU9Hn+XvSb
cRDnj6zW2I9Trjh/Bmy8QL7Ei/R/xbAfj37/IY5UXbABdTInCP1qwr6TyL+x
Bs3ruhu/8xXIDc1vIXIxxU3sd+56lHx+S07Remfd8X/H4xr2kSSh7y9lC/tf
np5GeSf3OXeY1muIon3I3+C8ofUvIjU6hQn9xlj/bDrffBC9j+a9cdZM4+HV
VEfaEUB+fV434flFWplF+avHn8qg9+Odd9F44sx8fH5XL1LdpHvFwhxYPIxy
W8/YmUr7XVOHPE8cjX1dn0z3Z1oTv1/4/JbN1WhesN0WWsdx6AZ6jv4PNtP1
T8dk0fPoPiRd6HcYQeurLb9MJi3MyaT8T+iE9X3Td9DYee4O4f69fqF1Ve9i
rN/+Mp6DX8FOfI/7pdG42Yl1ws8vZGk0zfskkO5fXutJ96O6OG2k+Vemwt/3
zgt5fjM2u96h51NTfJv217KR1DjUma6rjUWkhtmzTnj/r3rTfN3yGan27nBS
afxS6Dl71KnIvCPs3+OzkS8JZlItwoT1B3tiP4O+Rn2tndAvj3NBvlgikC+O
Z1BvYivyKfUuqHPJXuhXL7aA360HqbLoe+iK86izyuYfGXlb6M+oo1wwPq+s
xXN0x334ric1zuSSygeihH7TkcvwN5yHv30z+L4Pwv1sm4V6dteF+ancPo6c
XVEADbDl9sgG6PI/kO/BxUK/KfwA8nXMflLZoqNOGcZaUQn253tZnH/Z+P2Q
R+agTjhUcd2DOh6op68qEfq1hHSsU/4z9G4G/JVQU9RurJ98XOjXy5Mw/04q
1umyDfM11NN7Yj+6/xHx+XNKIq6v34z9OqZAzagnO9juy8Ms9Jtq16J+2Sbo
8q3QRfBrk7Iwbjws9v/LqJYG6mvlndfqaZ8B9TRWV4+jsTHWQqoGtGkQPr9o
V/h/mkkq1Swk1Y5lQAO8ya+1ulEv9OdFUF9qe84gJwLMyKOlTtiHbyypHDxA
6Ne6DkVfl04klXIjkVcuaRgXdoA//IEwv/QhzvDb9cc+wgdj/U0LSOXmzeB7
O0rol767j9xLqkFufdEe66/uiboJO1Fvlp14fftr8GWVIS/Hmmi+Xod96atD
4R96Q5x/Rcfhu1pIqr5+CnnZuwC61x91HMKFfqNHPtbPz4M/UEdO5RzAfbW9
h3Hb6eL8rMrGPH+ovj4duTsqE/V8SrC/dsXC/FMSU/F+nzTML01BHZ/tuO69
B/XMBUK/FpEM3yibXtoK9UQdeTL2pa6wCP1G4Qas42XzfYSxEoOxZM7C89ly
WJz/MzBfnazh/TKbv8FWr7tt/bZiP8MwzDPxJ5lla98=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 105->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHlQVWUYxi9uIRriMmqKdRGRXJrUHJVJh6tCJiIpmbiUc5JcWTSD0hHl
yBDqKIxOlihpR1AhJJVFXBA5oiIoQoqguHVEwQ2V7XavS2Pe96F/mi9DrGmZ
9zfDPHPO+Z73/c7hvs894DB9nveMpjqdTv/0x+7pj7WOYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYZj/Gpu9z7s+FcmnXZlrI+yGuORCi89wzrFE5JeNpZca
UlfquDqnMf0ZhvljtJrMCtFcqevCb9PcT19UZVHFdUWVaJ0ydOtJ4VxOLrtg
Oa/v2OumReXsPKH/zzAkTCim/Ghrc78xfn3lOsoNNdz5TGP82pbouCyL/4LT
xRfJH3lNi6+zXsDPMH8HSrIfzaUaNp1UGuRcTcdDdKRKyWNcT/SsFn7+T5w8
S3O+IfIKzemyM99aPueKuQ29N2g2ZvhvdxH7fyPWZRxddx4XTX3fqd5AuXH/
5XN0XGD7bP/vUH0nrKb++Rk/UX/TnefLn2Fv033ITq02Up0pLRqVX2qpSwz5
ByVlv9D7S+TEaZwfzF+NwWoSzZVhzzhSzZSI+e93BNpzdA2dz15YI3xP6B1G
3//6G660Xj66h+bE0HcU/GU+pHqtk9AvjzyUSueTghJpzj/67BjlULhUnzuv
kl9aE/XM+Zfcpqyh69lXt5B22H+N9j0pkOroA0sblB9SZp/ltC7IHGuZN21z
AN5/KiMalj+9W4dafFJHG8WihuAllbSPHsUNyg+t08QE8o0+HkbP0fRLEfnu
Pip+rvz4/ANvqtN590z6vdx78H5j8kf2bBtEz8+50o///vr/ocxKpLmUXcpI
9d9lkBpOH8P5vNBa+vzG7K0V/v5trOG7E0yqGF/BnB+DaofzcN17ltCvDDlz
l64HrcScHw7Ae8d8pT5PinD8JEGYH4YOyfT9bvCcTXXkHA/kWcgK+LvHIn/S
xwr9umsbaL4Mm3M00nJ7+HuOhy96Kea+/QRx/5CSU7TvVTl4DzreH3/vpJfi
fWobckPbdEeYH7LXIMo7vZcr5agWu+QqHbukw+9/C/nVrpl4/63M+ygnL2Vm
0brC7Xm0bsFCqiNfrEXuZPmJ82f/0DR6vqHr95DmPjxAfWeHUR3VPJf+7yNd
andB6J/ptoP63M7bRapup3ry6AH7qd6CXnR/ct1A4fuPOmUM5bU8ziqe+gUH
x9Lz3zUGx/7N6P50/W3ThP6FZvreMIzwhK56ay+pRwz55JEjUug41C5F+PzX
O6BPaBCp4u+O57B8RDqtn3FwN9VxWqQI/T0HrKXns/jETlpXdIXWaady46iv
1Xny69su/ebfmJ+S+XUj7TPxeh3tL+8CqebmQOelknOkarTjz8Lnn+hO69UV
fvBfDCSVK1JQJ8Ee9UOOGoXPb1ggcuFRQC18q0ilUddJFdsoqiO1by70a1pX
WmeY2wR1PHpBu32K88OMyC9vU51wfne+hH6+LUnVle2wjwde6G+swXGgr9Cv
dzPRXKp21aSS1Wvov+Jj+GzSUafPHHH/jhXIyYDLyNvqKsy57Rjy6X0i4F95
WZifUsWP8E0tgG/5CeStsRx1bz7E/k7ME/szjyDvp2bCN+sgjt9TUbdzPs6/
eUf8/heO7wvljX1Yl5qOfu57UWcg6qg3S8T5FZiKdcOTcf0TqBqdgnotD+C8
3Wmxv1sS/GOhWkY8/Gd/wPrB2I/25KTQr4/A94rULw7r4rehbzK+F7Xxabg+
Nk/cf+IWPKfmW7GPuu04rqzf1yY8ByUpV+z/h5HMxZjr209IDa3bmmj/kgup
FFxIqlM6mYXzO9iefGqBDzQqBOpQSKoc8CC/1veKSfj5KYhCvvSogJbdI5W7
9ia/ZpdAKh33EPrlpr7Ily+DobXx8CtXSbV0F+xnTxuhX3fQEX1HBcAfuZhU
vymVVLFqQn592VfC/JMnPMBcx1SR6h/3Rm4+ngl/lyzkVpmd0K+zvoH8bGZE
zvW1gm+aO/Zjmoz7OGgS5p9acgZ5O7wI/rXF2Ic/8lwr94Ivf5k4fxfmov+t
HNzHu9mot/E86s3H/pThX4j9TvuxLiIDdQan19V/7tHfqRT7uV4izD9t6S6c
j0tFnzn1x7fSsA/rQzjOOC30q9Hfo2/3RFz33IH+h5PQfy72ZXDMF/tPbsG6
8Hjcx6Y47MMe9ZRI3J+uQOzXW8Xi/rpA5Q+3Yl059qVzO4D6xblCP8MwzHPx
K/FldjE=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 106->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 107->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmAlMFVcUhsfYxZWoCJVU7KjEtGqtcSsaxVHc4oZgG2vb6CAIGjWCqRGM
lVFcUHFBayxKdRQE4QFCVJRFGUBRnoC4a1E6LlXRooLiUpfWd37StM3VKjap
Tc6XkD9z5/7nnhne/Wfeaz1hutfEupIkyc/+mjz7qycxDMMwDMMwDMMwDMMw
DMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMwDMMw
DMMwDMMwDMMwDMMwDPM/w+j36GjfZ2r6BZ7vWwu/3Gh0js2nlYccFvnlhNTS
l6mrLPwyuzbrMwzzfJS+Cy7S/kyO8c+2Dcy4ucSmysrW12g8a9Ftm+px0L9j
eizY/5fx96fNsx0rXZxpX2u3f7pqU8Nxp9D/T6ghvU9QvafjK2rjNyvzKH/M
cy331Wr9hlneNp+8OfTka+XPbyNVzi/mTUOp0GlfGoMbkmquvUjVrqcxLl0m
lRO6V4o+v8YF61naX+4uZbRPPWOjbfmhDgujcaMqH373hkL/H324Bc2n+b4z
A/88z2ymn6bxoS1e6H8e2nof6kvq5PJKfjVk2gy6roqe31EuZjrVKr+UD414
m9+8tDbvdfa/7pb1PecH82+jbwqnfWEO8SVV44JJlcU/kBqlzapo/7oEVYk+
f/qC+yWUG0mrkB9frUedVW6oG70Mx4EDhH6pQ4PdtM6smQk0r/2SA7TenMHY
rwGoo9/c+FL7V1ttXU/1HqZfJt++68gxh0sv9OsZlgC67gIvP1r/onsM7fui
bOSh7/yXWt9ofHUZ+ebmR1Mdr/7IjSXWl8oPdVAl5aDidTCZ6kSuOET3cd3Z
s6+y/5WCkjVUpzRlE/UxKWVzrfIj3Yo8H/VpVHZt/MwbjdbzFvblil9ItQ1x
OO5f/w597tr7kmqep+8In//LP4CvbTypaelPqtdfjvGeTcgnxwcL/ebkyFt0
vk8ocqLpGmhUHfKrnq1Q55pVnB+775h0PmgI1ZHyhiO/kpeSam0akU/xCBX6
jVgLfb/Q93Sg70FSeClyzCGdVCkupTqyxVe8/qM+R8i/f+Yp6ndWi3LyDfLD
e8/ODGjL48L8UHMH59N9S4ik30dUj+noI2AK1r9b08e4aGF+GG+Nz6Dx5DMG
9X1hj5Xqdd5AdZTQtVjfLkzolzsP3EXzEkvSSHM3ZJE/x6mQrssYRTkqbU4Q
/v6jhd9LpPGLpSm0fvBRqicNS6G+tPIquj6p4JTw/UfXh22hfM5Yvo3OO1VT
bhrGr3Ss3/g4nfo5EbxL6E+LtdD8ERNIzU8y6XmiJI0jn/Ko7Q6q0955h8iv
DPWNo3FHR1pP89ZxPwLO0P0wvZam0rij/xbh/z/veASNH3DfTn1MikTOzs2M
oXXHLCa/cc4z8k18fzMj7Kvp+kYW3qU+N58i1dw+onGlzCCVOja/J7x/BcPh
S/oG6jMGWmaQqglPcFyZWS38/LkuQz60sCJvEt/B+pk94FsZgTpjW4r9TWXk
yuOVyKmBBupsbU4+/edWpNI4O6FfWoN8UgudUGf2IlKjohjHk9uQXxmYele4
/zyeIKekKuSenT3Wb7AXuhP9qGPDhH4z6Rp8/S7gPWtWPcwv88P17ApHnbiH
wvyUnp5ELlmOoI6lBDk87V2ar1ic4b9SKvYvz8f8y9nIyeNQI+AAtNcV5G9F
O6Ff8d+L847p8DfcjXp1M6FhOdBRZ8TvjxbMV6fuwPUPhWoRNRqPunLIMbH/
YSLOq0no9zM8h/RyHOtT4NfyDgn9snPN/Mit6NMHzz/ZOwF6Pg2+TgXi50do
DPq3xGL94prnZ+9k6HBcnxFVLH5+/MeY0Sbta31dEfZ3u8OkRo7DffRfQqrZ
tXkgvH9TW9N889Q8+AtXkyoFm6CWr8mvOl+6L/S3noF9WZkBDc9F7vSvj3ou
s9HPLRex33M0zdcn+sA/8ltS40Ea6tSRyS8fOSbML3lqF/gCe2O+/Sj4s6JQ
18Uefm8foV8tQr7JXk7wt3IlNXt4kcrVqcjXw+8J/Wa36/B3f4B86NYY68/p
TKq+PQ39+TuI/T1PIi8fHoduPExqtr2C/EsYgT6sE8X5+bgA68YcRB9R+1En
90fknnMVdOFcod8ckIX1gvZi3pY90OkZqBN0Gvnd6oww/6QGO3BeTsP55shZ
sxrH2pRsnL9xQpyfn2/HeNeUmnWS4E/CuP4IfemxVvH6W+PxnBmE+cqJbfB/
kYz5A2quq2Ox0K+Xx8EfHgvf6ppj10T07437I6UViddnGIZ5FX4HHn95vQ==

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 108->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmA1MVWUYx686jVK0dEJa4UEIcQSMZOXM8AwVp31oxHI6pIuaOlsJhS7H
hwcxMC1Ft1Qw8RAqXAFBEUg+DyBQ6gAhJBHGUdD8IFHAz0KC50+t3JsJtfWx
57ex/+65z/953/fA87/nYr1whec7AwwGg9T183jXj5mBYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiGYRiG
YRiGYRiGYRiGYRiGYZj/GHJcYPmULlWT5tVO6YNf99qc3u2TfH8oFPmNjW0P
1Vf6Rj3Sl/UZhvlj5Ha3s6K50pttL9DcT/C61q36vg3XRHXqmqDfz3XAuOe6
XyuuJpprY9CB7+m1e57Q/2foy7OryO+8sbkvfuMUD418iYu39MWvBif55nep
9sWcKs4f5v+G3hZIc6kNmd1C8/pBOL2WJh8mVT/rf71b5Wi766K/f+lOFM25
NH1bPfUJX5nQPS/yhRfOUG7k+lMfeeYgof9XCgYr9P7ECbbUb9i97bR+nnqa
/PbWD/bff67QimCqH+LUQP7Qa73Ln1BLb/LdcYmj+S+716f8+gX1mGfBX/FL
VnO3cv4wfzfykeM0V+p38zBfPpGkeuVRUu2mYytdtwtoFf39afGXKsnv7Y36
rG1Q8zrkiE8s+v0kC/2K++4sqtvpk0r5M/9iKWljGHIjJJXUODT5gfOv+/Wj
z3dln7KT6lzcz1PfphD0mXzuofJDs4hcSfPesdTUrUpoMXIwO6hX+SO3n42k
+1A+nvxK0pWHyg/p5Vq/39Yp6XnFtP6zB0/3an2T9cd033Y0RJOO8dzF+cHc
j9xeQXMp18WS6oZ0zKl/Ia7bv9JGuq64Tfg9oWIY1SkDlvbUv0hqHBhDKvl3
4H3bAKFfWuWA7xeLmqG7m2nOjPGNpNKy+egboQnzQz+5hr6/6CGWeI5pLMC8
+5xC7nSakU9NSRD65RnTqqnu6YvnaJ+yOebcI4xU2RWHfJyxRehXa80qaL3E
cTW071EZl6hfWRvOU+iPfp/mi5+fttWVUF1g0QlaZ7Af7cNwugV50bkT+qa7
0K/PGpVNvreC8Hyx7/xx2s+qXOoj5+B7lx7VKMwf5ZlS+v+MNnlnJu3fcXku
qbU97Udrimwi/bS1Xnj+4oJkur5jzkHah9uiDHpdsIr2JQWbSvA90LdIeP+n
Xf6S9mkz20T3Oyp5D613tp5ey5tL6P8+sp1jhtA/0TmJ3i+4m0i+2x10Dvnd
TPKp/SPTqG++KU14/pb8ePKtWZdA70dV0f0wbraiPuqrbXQuedm8OGF+JvjS
c5ny4YYU0r0tsaR1Y/dSn7l15JdK06P/jfmrvTH7Bp0/9dt22rdVK9TTha4b
hxdBI56/Kbz/wxZRvXFpCKmcGkqqbSkk1Uegv/o++tyPmhNCuaBezUA+hBwl
1Wc5kd9wNRr9bZ8U+g3TraheqXBDTr22Gv5YE/qNnYT9xAwSr//2EKqTfC3g
H+gBnw32JYdbkl9yCmgXnv+le8gFqQXqZQnfWiecqzMSOne60G90uIJ8cqlH
bqY2o8+Y0diXwRb+kYfE+VlYTfVa8EnkZFQ5cnd/Lfp+3oBcz5HE/pWlqDdH
3qvhUMPIIuzHpgR9Cy4K88/gkI98n5WN+pYjPZ8juC7Zo58SoYv9S77q2R8+
dxS3w/DFZGJd1zz4ppYL/UavFFx3PIQ+JUnosz4N13OwH82vTJzfriacexJU
OZSIOodU9FsPvxJwTPz8u3ov9hkIvzwRfsXnIM4RnIX7UlkhPv8/jNqvEnO9
p45UPn6G1HjX+RadY04ZqbrA4bZo/3qMTY8/GHphHamWnkmqRniQXwpvuCX8
/XVuRD74XMZ8LmkmVZ6yRb+szej3uqPYH/kefKM/Qo7lJsMf1kgq93fBuWpu
iPPLyhl128OgQQugdlXIPfPh5FPi8oR+aXwn8mGCBdXrG5bgPGuxLzW2EP0K
7cT+xZeRk7M6kA8Wj+E8jyyEz8Yf55HHC/2aWTVy16wG6gpVbG5iX2P9cI7V
W4X5pwWdQD7WfI36piLk7Sd1pGrrDWj0JrF/ei58Ug7WD8rAOWZmoU/xlZ7r
Q8X5qx3G+qMz0cc7Fb6iNKh7Afw/lgvzUznVUz8yBX0cDqB+RY/GZOP6E2K/
dH4/3q9ORp8R8fCFJeE+XoVfCTghXn+qCb7t8OneUDU3EefZhPsiPVom9DMM
w/SKnwHVs2OL
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 109->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHlQVWUYxi9iSTOWqFmjFB03otxyyV24JmMziWKOUqbi0QZ10iTHTG1E
jwgCSS7MCBOgc0QQBGURAZPFAy0yCsomCCZe0EgS5Moii6B534eaaebTEP/I
mvc3wzzznfs97/d9x/s+51wHLneb52qu0+mkh3+WD/8sdAzDMAzDMAzDMAzD
MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzD
MAzDMAzDMAzDMAzDMAzD/MdQg6vO2ZvU06XYvgt+ZXGxavIZ+qxPFPkV16jS
ztQ19IhJ6Mr6DMM8Gjmw/1Xqz6rB35hUCouYReODoTdIQ58zUt+57zKK+k/e
lpkmuq7uGU99bcib8ptJtVVnhf5/Ql9iVUA+hzdvdcWvVbalky/7o6AzXfBL
6bYq+SKl/KfKn8JBWzi/mGcNZfkU6kv5y2m1pHZzaKzf+Bn69eYdUs37vTui
76/m1L2MfCG3kSMNFZGmfpEd3fBcDypAHZceQv+fqJVTNvzt86vllEf6cd+V
mFRptHms/1FodoHXKNdSBz+RX9qw84DpHMqSJRHU/wV9u5RfhtUv7KP3p7y7
mU/T/9rnM7Z2Jb8Y5nEYJuZSXxgqU0nV1HhSeVEp+uWN4XXUP32+rhO+v/tm
4fm8eBDNVyqPk+oth6LeuI2k0ph3hX451C+F5hvN42ndnPws8oWvh1+2x/5e
UoX9q9bGLqbr6/ajz/oVBlKdlesqyWdmST5tbflj+1+29thB8xxXbKO+3zX4
iElVt2Qj8md1p/JDqZvrQ3XOtwWTb89E8qvfVncuPzLuhZrW1Tx37qY6YVfO
Up3FTZ36nfTXebwG0fuOdluKpjweaHW4S+9faUu8af0grx38/vI/ZHs69aUS
co1U3qGiT/saoREz66k//X6qF/bvgrfr8D3dSWq4M4FUnTydVD+zHPXXewj9
6mvueP/45H30ubUt+n5oOnIkDvmkxH0vzo/+fSooX04vQp29He8ZziOhZieg
FgFCv8E/5xKt5+p1nT4fshT9Ou0B6i2LQ35FO4nzL3tpLu2vuQL/P7LA73e6
H3c9kRu5HshT5wZhfsiFi6i/1br2HPLJW2kf8qfbkRfdy6AO+4V+aeF5yk8l
rzqDfMHuF+g+hg+g32/SL1/B/2CvMH+0C45JtH7AqFP0udUM+r2kvthK+5HN
nX+lOilNZcLzt++LofV9z1F+60dUUz1dmG0qnWe6M51P22rxo/D+V/lTLunn
DImieRc3hJNuuUhjKcDlNI0zjEnC8z+feIw+77ecVK2cROfQLlvBt6b7SdK6
t04K779Tz0i67tHrKN233ha0jly8Ipl00vITVHela5j4+VPtT9enlsTSOWbN
D6X7kXKIzmE4MJ78eoNtyLOYn+pUYwPt72whqVRZSqp7fWQj6YdppIYBw+8K
79+0IzRfDp8N38EgUm1ZIq67uJBf3ljUKPz++G2iXFDdk5EPp8pItfaFqFPl
R2rIsRP6dQE29ejPuaRyazjGA1oxNs7DuczqGoT793sZ69+3IzXE+2D9WqsG
5II91o8OF/vDzZCPeS3IB2ky/L5BWL8tH+P5zkK/VFGD9yufG8jf7u3I06DZ
OMeYUFKloVGcv8HFyF23AtSJu4D8jbmP/YwdhfXt/IV+fWMW5jdnYv61DOT2
qh+g9fWo7zlS6NfVnsF+tVTM6wWVyqHaddSRMy4L81PLO435UhI+H3MKz4sy
5L1ipkFtioR+1esk/GEJ+Lz5OHRWIta9mdJxX7LF/tmxOL8VfNoXUGl7HLQn
9qGW5Aj9+vxI7G93NPzx8CurT8Df1nE/jl8U+v9tlFeM1Nfa0BpS1awNYwfn
Jvr3VDNJ9beGNQv3n/IOzZfaFeTDHh/4R19BvThH8mveV5uE3181Avli3UKq
T7iOPu82CfVqj0H1E4R+LcoN/qjNyJm2dNRRiknVogXwr7ES++9L8NVsglqv
bcTzQkXdlH7kl6NChPmnX2JO85SxlvBHOWH/DtiPOgLn09ysxf4VNci5EQ9I
ldhu8H08ET77laj7arU4P0OKkU+HLiHnbEqgvreQv0U4n25zsNAvZeVg/Zbz
yKeAn/E82N3xHMipQj1/K3H+DkvHvltSMb93CsaHO9T7Cq7f7in0Gw7jOaFZ
JuEcifEYByZjH3kZ0IQiYX5q4+NwzlGxmDf2GOpUYWzwx/7ke9ni/P3gKD63
icF+KyNQZzTGcinOpaTliv29InG9BD5Fh7GhFfvQtDTUkXKEfoZhmCfiDyNJ
ifQ=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 110->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
GIH+ykCo7iuo7ILxFP90tK+L/QzDMI/Fr1J7a1I=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 111->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHlMFVcUxl8guGGDNYq0aWWavBSXKFq0rVsZrZpaF9AqirFmqnFDDLVq
qlF0xCKNGEldcGt0ALEBQVBARASGRRSNsqiggjhPAYu0iiziRq3vfPxhm9tW
aBPb5vwScnLv3O+cM5e53wy8M8dvyjxbk8kkPf/p8vyng4lhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhmP8Y+pMtWe7PozImrdC9LQliniikb67dmSG4rC1Z
ePVl8qq5zQvaVJ9hmD9E/dnuiuhcyXlTLNZ5KT78rjUaGf61onVKcEbqi/Ny
Tug4OudDThVb5+WaBZXWqAYHCvV/hVwVSL6jfhJS/SrOv+GU8i3Vvz3n1N+p
b1RGBov8j2FeJdr0GXQu5bX9cc4nL6WxtjwUUX1C0Rgx+L7w+fd0NkgfN7vs
Nz7gEERj7dpt0iuO7cX63/fzbMPGF9dp0uel5EO7za3yD7n9d0tIN+16OfUX
sqd1/qN7LaNz75EdQ+e2Iulea/T65dV+VPehe6xVr4ycltMW/9AuVvjS/iWN
28TfP8w/TuMunEv3aRSNA+Mwts9CLO9VR+do+9d1oufPuLTtMj3n4a60Xq1M
RNxqC70lBXm/8BLqpZU3jtPzPTgznvziw91nqV7wDtLJyciruIUJ/UO3ZIXS
fOHYldZzpoX/tIbqn+v2ozXqfVr8a17mn/qP7tren+qbx+6g+jbdo635pLhM
0ktHfF/Kv2TvQsqjdihYT/ly50Pv97h1/pO1LpD0bp1Pk67bg9LW6JVeN8j/
5A1uQRQfnd3dFv+Q9vaNtO6DEbtqH3+//P/Qvkqnc6mfz8Y5n7kV43dvUpTt
R9TT82zJrBc9P+rjgbROS5tJ0ahzRp6pgTjvZlvSyXWqUG+60A/n3OYIzvn1
SPiFXyHmM3qjn9qrQv9Q4kNu0vUaB5zPpNM4bwvWIM9HFoqSxxax//QdUkLX
9acVVH/nJNLrJ9Lx3eIUgbz3Nwv18jeu9PeJNtqC/2M4+tRQvq7vwgcDlqIP
o4u4fhf1DOkbHPPp+vg+1Ic6/yTqx2TBvwqjxf5zZVQazWf3y6a+dx+/QPWT
QyiPlBuG77vhutB/9ColmXSeKSm0rqeDTrp1efn4vVdX0bxdkEXo/++5kG/L
TWeP0n0MNJOfy/VF6GvgB3k0n/+L8O8nxfe1SHwn5R6i+83p8wOtSy6LoT42
BadiHxYfF+7/5o6H6fqKgli6XlWE+j6WE9R/2b1Eur9jk5KE+t7tomj9+NnR
lCcq5BiNq9shz0DPBJqP9j8o3P8zPtsp/15P2gepeVgEjXtH0XpteD3ti7yr
+/5/5fdbR7dG6mvtzQbarxJ7Gmvl0yiq+jaK0sYBD4T7N+AE6XT/PRSV0gMU
5UUlFI3vJ0NfmdwofP6Cd5IvGFp/Wq/lmaFb6o9xYgLyuYwQ6hVfmfSaQxL8
5a0q+M2kDujr4lr0NdNBrLd1ovW64wLEPfuRp0dn1A0cg356HGwQ/v7G26De
2Eb4lGko+pm1Hff16XHkveUj1BvFd+Gzs2/BHyY1wz8jhyHvmVnop2uh0D+V
yGuoe6SYouJXBL2NHa1XIp0pSvcWCvX6rnPQrzoFvROiFIT3gTy6oiW/nVAv
3czEutqT6L8z3idybkucm4EYUCz0P20OdJJrMqKMaCpLQd1h0Ethl8T+PzgB
fXdNRJ0JcRifTMJ6B/RhWC6I9RMOY75HLNa5YKw8PIq69mkt78V8sX/XReH9
5x4DfQDy6B7ox0jA/Rk1BUL9q0ZNvEznWsouxPke1UTRmDioifbBHE1R+6z3
Q1H/arwbrVfLv6Qoh0ZAH38O+fYuI72x/lGT8Pc/KI3OpeFRBL+JrYP/1ExE
vsn7KOohQ8V673k410/XQz8jgqLuVY08r3tDv8gs1BvLzI2oMxc+dW0x+gk/
hbwNb0LfKVXof8YEO9QJ60lR6TO8xS9DkDfgKvLFm4V6NaiWfEFKaobPeHVE
vjoP6N9Yjbwzewj1yvJS+MpKRL2uHPlS78DP+09HfZdwof9p7vnwx+7noT+T
B937xfA/4z7yr5gi1JsWZaLv+elYvyMVeRZirExFP/rht4V6NSAZ9T7Ge0S5
Bb831GOIfjl4D9hdFvunJ9br2Udx3yPjoBuKsVGpY77kglCvdzqM695Yb7p/
CPWWYqynt+g9Lgr1yjOsV7IQVWdEwyUe+kfQa3cKxO8PhmGY1vArhvR4CQ==

                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 112->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 113->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztmHlQVWUYxq/i5HVfMpfMuISTlVijogFuxyFpkVwHEc04QW6RoWaKmHpG
RBIXQAltQLqFYaLEIgiGyYFY3GIVdTTggAIiFouAuJDJ+zDNNPNlCn+0zPub
YZ75zv2e9/u+c3mfe+41c3GftdBEp9MZHvz1fvCn1zEMwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAM8x9D62l7aFKLll1KndSeOgUfLxL51e7fXGxPXYZh2o6W
ejpH1H9G86VFLdcN3i9Wt6hRXlkjmqedMD8iui65WGW2XFdKxl+hOv7nq9vS
58q8vXlUx994tT05oZTaRyW3xef23voWn+JrFduu9ZdYbOOcY/5taHUu1Ney
afMv1OcDXGlscI0hlfRl6Htri1phTjR1K6XrtmMLqc7aTZvoeWFgnNai6sHz
8C/TC/1/7GPriM1/et2h+TD55y+mHDJuNXmo/y/x8KZ9SAO6C/Pr75BXVwXQ
eQLfeSy/YcWrAZQ3Cc57KL/Cbb5uU/7pzgZT/jz78+a25BfDPAxltBP1lWq/
nNSwbiap1HUvrs+phn7qUSf8/F9QfoGuu9rSPGPhGlJ5zkek2qklGJdVCvtX
W9yQRK9PfCKG+uTK8WxSt4vw/epHfWe8/rY4fzzH+9N8+5nBNM//WiDtX5tR
SWq7iPzKuA8fnh+dRqDPo3ZHtvSZ5t57P/kMzej70l2P1P/ST9120j7mZfq1
1DE6T8f6XbY9kl+NzJhP+/ZZ7U7+1y3TyBd4suhx8kPdvsaS6pR5OVF+lAds
b9fzx8n8ufz88j/kQAX1tWFKGKnkuIbU2DUfutDlJr0+J+Wm6P1X1t1G33e2
hn/DU6i39n1S+eA1aC8Pod9Q5YA+12Wizwsa0ae/1SCHUqai7uRQYf7InhH0
/UK+fQjPKxt88PxSfgvPNV6vkU8Zul7o1wWl4feHTnnlNG8L6uhW16KOy1X0
b6Sj0G9w1NP3EzlMu0z54dx0g86Rexk+82icz7tEmD/K5YhTyKlo+h4mdehX
Rn4nK+Tn1KW4vxtnC/1y0fkTuF+bKSekl7JQ50AM6qwqwn3QCoX5o870TqR8
vzYDOTyqUwr54udSHfWLHRVUT32mVHj/CvpTbhvfraDvgUqI1zGaFxSbTPfP
ah+dz2C+PEN4/pDEcPL5h0TSOoNND9K444Ao0glFx2kf1kXHRH6p1+c0z1Do
TqqFYn2lcyqdR32rJp7GC3yPCvdfbR5B6+6spN+5lPhRCTRveBLdF13wlDg6
n2nEt8L1S8yD6HpzSjTVCZ1Nnxtql+IDtK81Kt0XaZfpV//G/NRmWTTQPp21
ejr/smJS3Uj7Buw3jFQOfq5R+PkbvY3ma0+GksqvREGbk0nVngvJryy+0CDs
H1sV+WIyAfPrx9Xj/2k91C0E9Sfaif2bVpBfMx+EeTZ3UK+PHY2l05+gTu4Q
sX/B08gl72nQsz7wTx0GX0cH3Bf9kXrh+3dGT/ON1bXIh4r+8DetJFXvZZEq
GV5Cv+QGn2p5FTkZkIe8u30H4wRf1Os8VOg3TCyGfxx8ij4H/vQs6JvDsI+q
cGH+qg3ZyOvJmch7px+hO9LgH1OHc8V1Fvq1vpin2KnwhUK14mRcX4V6atkl
YX6qJT/g9eAE+NO/x7g0CXWCUzHOLRD6lZqjmLc7DucwicZ4RSLuXxrqSKOz
hX5jHuarY6Dy/CjU0aOeHNK6v4Qssb82Eust+w7nzW2tk459yR1SsH71OfHn
zz+MEq5RXxsD6kgNd3vcIrUcSyqXHibV9RveJHz/PCY0ov/CSOWBfhivbURe
fPkG+SWb6lvC8z9fT30pxd4n1ZruoU8HLSK/tO8U6l2wE/q1D5YinzzPok42
8kqL7A6fbgmp+vJQoV85Nx7+GwGkRhtfjO+FQ3sMJL8SESXMP2laN6y7fBHW
dRgJ7ZgP9blOqvqNEvq13JvIKc/emB/UB+sGzsJ9KNgI/xYzoV8eXAj/C8ht
aWcV8veiHrm7fzby/W6qMP8061zk3Okc5Mv9DNRxrWzN9WroMUehX3FMhc8G
ajCewPolGCvTsS/VooN4fbNE+OQkzJPisV7fBNQNTcfrgQXC/JPVWPi2xmCe
RzT8n+G6dKZ1P9k5Qr8SE4nz18Nn3HEY84cg77X9OIe8J0+cv7sx39gVdXRm
GOuaUU8Oh1/T54s/PxiGYR6H3wES/Hfv
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 114->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 115->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztlwtMVmUYxz+0vISa5hQlyqNpOhUb1kwN4dg0TVIDUWwtON5IxEict3mB
o+ZiOWVCoHM2jxoRCAiKIH4iBwVUULmpoHE5ilphiCACKkXy/L9aa+/cgFqX
Pb+N/fee8/yf933Pvvd/DgPnf+q2qKPJZJKe/PV88tfFxDAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzDMf42CRC3tieh9i79xbkcbTXnfVeSXOzUWtacvwzBt
R54Vfkp0/ozd7ldbrutup+62qBay557wnLq+vk94rj02UF9lnadBfbKt7rbl
nEtWn1wgf4KL0Z6ckBzzHzm1wae/fGNmy7zyGdstae2YX30xu11+hvk7UH3X
0Lk2ttlX0Xl16UNj3WcLqVZVAp15R3z+X/OooPrVXUv+eF+afPo6jRdMqKG+
5ZfFfgv60sWr6P6OmDA6b/MjI2jelcPLW1QaN+6p/j+jZcRvIt9Mb/IrTinV
rfGrHRRaj1GyLKTl3KquAa2a//c+pfLGFr82Z0Vou/Ira/sG/k5i/mrkdQF0
PlXv9aSmbn6k8ueHcG7nNZPqB+bWCt/zUztfozrbGejTcRGppi6FT9qFfu7W
Qr9U+EoK1d/sH0fnzXZePtXZ7MX8VwaSSlcja4S/f/9tw+i9euXZzVTXe8xO
Wsf28ErSrO7kM15dK/ZbUN5yobzQHlYHky8uNYbGs3phHQN8n+r/De2Ka3wa
reO5UNIiT+TphPxW5Yfh6BdI63jGhb6jjDCb8racf2XiGvr/TbF6c1Gb8iPH
7MO58/9FbarCucyuJJV3p0Ob+96n390SF1JtRdR90e9AuTiE6o08f6h/f1I9
OIpUs+tOPtnBT+iXHvsgN5YmISeGdcB68upwbkMX0lhdd0uYH0bK6Ft0vxHn
Ww0ycN4cMzBOfh6+FzyEfrVqPeWXtvHR9zR/TlfM2/Ma9ZHK9mFd08X5J9Xd
K6D6u7PLqC4qgP7P0e1+xneVGfmnhZcJ80O5UJdN6xx7FrnX/53b5C8vxD7q
8N1lJGwV58/bTjrlk/d3mfheyaU+mima+kihEua/PV7s9zJT/urr60/Q+u06
U97Ie5ZTH905+AfyTfKqED6/ub0PU51n/6M036YuZlpPUA6tS4/wy6Hx/RFn
he+PYz7f0nwRAYfofmBRNPn8T8TTuGTySdpX6gdmoT92K9XJtsGkSs7s47Tf
rWbaj6l3QxL1y81LFvm19DEH6X4nNZbm8S6jOvl8ID0XeYIr7Utvvh8tfH5T
ztL7Rsken0D7HxpI363qKIco6tclOJF8czL3/xtzVLXKraP1F5WQ6tbXSU0V
Qx7QPj5aCh3Yo174+z8fQvXSrsOkRv126JAyXI8ZRH6jKeKB8Pff7SLyxVum
enVzP6jHfPirVVKtyFrst16BXPksF1qcSqqsvIfx5rHYT+fiOuH6g+yoTt03
C/WKPzS7gVTvO5x88pSJYn8va9SlNyI3M+zR78Be5F4k1qUN9RP6tYpa5ErY
j/D3a8B42jT0HRxIKjk3CfPTlGQgb62KkZO5lzCOQp4bgb8gh4euEvq1BbmY
z/EcqRKTCd2IsXQxC31XPxTmn+KDenV3GuZbDtXzoZJxCtcXloj95yw+72OY
txQq+6Wi3uE0+uRfFufve0nYX8gRy/7jofboYzywrKe5QPz+uIZ6vSYWfWri
UOeGftKXJzEOyhP6TWsPYb3uFp8Z/Uy+lnWtgF+dXij2/9Ok2DXQPqeW0vlW
HG+Sqslz6bqhJZLqmTaNwue3xBm5YPcF9NgmUu3MeVJjpD/83jUNIr/e1QG+
iT/R+dbeGERjOdifVHc6iD72o4V+7V0dvtgYUrlPCsa21ciLul3o99JjYX4p
X82AL2sJcs68Ev7Hx0klr0pcT4gV+tWRPVB/151Ur0U/3WM/ci+2J55ruCz0
a0ceIC8ju2GetSMwr7MX1v91KPplNAnzT/+wHDlZalFf5Lc6tQP6eH6MvoOT
hH5jQAF8lfl4D+w/g5wKu4HrD20wvz5J6JcyM5HP5tN4f6Tp6JOSjnUo9djf
6mZh/qluxzHP4hO4X3oU9eUYGxewHmW2OL/V6iRcb4RPjcR7SA5Ihq/Ssq7a
QnF+ZySgflkidFQ8fDL6mQ5jf6b6S0K/cScO8ybDZ+zEWNlxBPW+mN+IFs/P
MAzTKn4FFelz0w==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 116->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
xSUJ8UQC6jxAXVPF/vLwM2I9wzBMffgddeZmpA==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 117->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl3lMFGcYxkcUtaIWkZTaJe1IE2jj0RoVU1EcWtQiUWtrPWqiI8ajIqKV
aDwziYpV8QqCR1o7tNogKodyuuCO4kE5qlVB1EoHXam6KFQuRU0r70OTtvn+
wNqEpL6/ZPPkm/2e93u/3f2eme0ZEv7xzLaSJMlPX65PXx0lhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmGYv2EkzLo0rLWbYJgXFD00f6/o/MmD+19ouq74
5VQ2qWprvPcs51QfdDK7ab7hWHWFfK94Vf6bc67ZRxQ0+fRh/pc5Jxjmv8VM
dKumc+oUS+dTr3ufxrJ1F6keVEYqdW//m+j86d737ZQPg49dpfe9ts6zPRXt
o7jrTWPzpc7kU8cWVbfk/Brrtc20/rpJ2yl/khzXqI+Ygy3y/xPFq1sZ+aWx
d58nP9RSa1Vr5I8a6Bxqa4V1mRcD/bFG59PoNo9UW/4mqfLGPpzbsEaM7RPv
i37/2upJdD7l3oPh10bC576YVI6Nw3W5Rpgf8vpQG53PccczSL3bnqd+7Nto
vtlvNPwBqtAvjXTsoOv34hfTOfF02k3rfxLtoOv5xyk3tI5zxf4/+3CPSiR/
1arFNL+9czzy5xD5zRuPWpZfpVcXkG9b2t6merJbDHI0r7Fl+bXzdgitXxuw
iPZx8TP6fNTUbb88S/5oQRvW/XW+1idk6vPklxz34Qp+/vr/oWln6Fwbjw+Q
qpvW4Jz7H8bY94Ma0qJNNcL7/5OB8K+Phlq94PfeSWrGVZEq6nChX7Z0oXOp
d72L8750Ps3XSvxQpyAGde4WivMn+mQF9bfyJs5XvYnc+WIyfKMXoK9ITeg3
Mj0ov9TNy+7QvD6d0E/6Gmjgz8izIWFCv2RU0P8kyeMRnU/1dF96TtDD8byj
T02A31EnzB/FElxI+51tp9xTKwppP1LjHexnZT/k88FIod+Y4Hmc/Of8ztC8
/FepH63L5F+p3nYbnudiXcX5l/v4KH0//lHHyGfplku+cgv1o0+7fJs+x2vr
7cL1p3scQd/B6bR+8DT63yetzqS+zBkq7U/q5JIv9LezJND6DQnJ5M/OO0j9
RjhSqJ9p6ci/6+XZws+vbxDNk8fMIjUNXyuNr2bm0HhM90yq7+eeJfKreQqt
p/ftnEQ++1qary8ZfhS/5wm0Lylt9CHh79+Yu4vWKU2m9dUB8d/TvJCNtC89
IjyNtMdK4f/s1kZd4llH/Z29V0v9/36SVBnpR9cNn7Wkpv16nfD8jv8GvhNf
keoFG0ilqjxSM3EG/FFpYn/DTcoFo40C36LXsH7ALOhGG+o6+wr9piWK/PIo
K6lWj3pS4ENSPX466n5r1Ao//wgL/AP9SJVlYcg7HzvGOW9hf1tyhH7FpyvW
Of8Q+XDKA+PGBeijqAb7O7tQ7J/ZAN8UB3JyD/JSmhoA3+ffYV+3S8X5W2XS
fNnlCnxDi5F7O24g92I8sZ/K6UK/ol7EvJQC+Pufhj/3R1z/9BLqf1klzr+Y
5vvH0BPI7bJjuG+Mw1gKK2i+v5SL8/u+gffdrNj/iizUqc7BdTMXvv0lQr9c
iPlGz3T0Pf4I6iRnoo8VqC9VnBf61Wup2N97KZi/NBnjUNTTQ2yod+uceP2J
uE9qL8MvuTTXe6e5r93oX30g7r+10Se5N1Bf+6rrqe+sNjRW0wNItdmxpHq6
zwPh99dOJZ/qSCLV+kSSGv4dyCfPH4r6va0NQv+gETRfsrrCv8ETfSiLSM1w
G6ky0FvoV6s0ygXFtQR5NScReTPTAl+XQNR1Ka4X5sf4MMwvi0a+zI4l1dbu
J5UPOMEXpAr9+sPu8PVahj58g9HHkAKMD9Sivvu7Qr92pxH5EtkZPvNtaEYI
+qiPwPhmB6FfWngDueJajryULpLKrzsj1xunYB9Fe4T5qbkVw9/2Aqk2pwC5
vekW6vV4Ak3rJ/QbZWeQzz6nSI2AE1g/C6p+XYl6qU7i9bdY4R+BnDe3ZKDO
T9kYb8V9RNl6SZyfvdLRd3v4zLwj8LvjuhKEvmTHBaFfG3UY8zLSUEdPhgY1
10tp3teAEqHfWJ6EeT+gjlaIsVSYirqPTuP9oGLx/YdhGOZZ+ANxQU7G
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 118->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
+UL49K5Ya+Xop23BdTFc8sXPH4ZhmIbwN6TnQ4U=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 119->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
sZ9hGKZB/A5nJWOQ
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 120->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
SIN2x7XihrqKH56PNuGc+PODYRimPfwJkZVZyA==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 121->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl3tQVVUUxk+oaBIMqImPm510Gs0HKEpmI3LKCbNURhTTUrwqaKGGmOMz
9IySyoCOaD7yQSefiEGIoAjCPYKoIYiKIJrmKcGElJcPYLDMuz5q+mNbwTST
06zfDPPNPnd/a619h/3de1+aGuQT0EySJPnxn+Pjv1YSwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAMwzAM
wzAMwzAMwzAMwzAMwzAMwzBPO6OXZ3o+Fj3G+5ZV1ZTPyzybUEb9pvYs+aJC
f2qKX9vhkEH9N7nlNsXPMMyTUZ433aF7XjSt1Kpaj+MVVjUnhFbSc0fbKlpn
3qoU3T8taEwx3c8DNpetKqc5T6Z9JcE3yL9HIr/ySoXQ/yTUsJ0bqF6E5Sr5
X97aKP/vyBb/K+RrX/BzY/y689U2lj8/qIu+0xi/Vji15b+RV/InGwKsc2iH
9RmWv93NMI3DcJ9F91N6ayKpfMAF933JAVLdvxLPHX2rRf/P5sBOP9C+zr/Q
/dQ2f4j9e+ahztCFWLsaVcL82BJJ3zPU9/qnWtW4Mb6Q7vv4SOTGTW9So+06
oV+3r/+C+oZaQmiOtXd3U7+V3cqpf00K5jK5Cv1/1PGMmk3+Zut9aJ56jxjq
/1E8ctBu+1/mj9bMO8Z6P+WwPlTH7Fr4JWkpfObiuH+WX/tKo611FMc3N1A9
Uwm9L4oSZDQqPz3iw2ju1bvDqJ6XaXpT8kOJHryA3v9K2Ye/f/3/ME+MpXut
Bu/A/fY7R6q7pZEa6a/epf/fZN+7wvv3wSjap+UnwLd9HKk5bxGpPK4SGu4q
9KtnnkXfQSb4Ywdg/4g+6D9nDNY58cL8kbIu0fcWozaQ7re27nXccwvqmFe0
xXz6QqFf9wine2WUTaHPd0ONxH2fhO89kt9t5E9XJ6FfWzuB8kquGfYj9ckI
xn0ffY5U8Z2KOhsThPkjF/fOo/oOIVRHCo6i80h7UEdb3Av5qXuL80spp99H
8syB2dRvsj3meWMd1VET36U6Rt+rwvxRM7ek0f4Ldsdpf/X7WTR/d5uLtN7l
cJvWI9sKf78p7V2S6P3KzE4mHXQlneYesZhyXWp+jc4nuc/JEfYPjYyl5ye3
JNAc/kHxtB4/PJHqzfejuQxHW114/sBT5JPadT2EnK0+Rvvf1ixUr+2IFHr9
eniqcP75s+Koj/t96qtGFh0lffEc7ZfnhtO5lJg2B4X9Yy9tp/1fJWGOYXXR
tN7iSefSSzyP0Lq8VfTTmJ+qbYf7NJdb3T06b8sTpObhPei5vGwVXl+ad1/4
/jWvhW9UIjR/M3R1DqlUNYl85rQIsT83DfnS8TPar9YMh89uI6kmfwsdUX9P
eP+WrCK/ej6TVBlrQ/v0Rz1wjnlzMc+8fUK/6tONfPIKH1Kt91bUs4xF308D
oeuDhH69XXvsP2sPvz/q6L6bUDfwEZ4v7yn0K18/RE5drCBVdvfDOexU+Pbm
QU8VCPPTHHwTOfnOddQpuoI8HySjv+8kzHNmh9CvORcid/Vc9A/IQV465SN3
1zSD38Ukzu+J2fD7Z0EzdPiHnsZczhdRt1+pMD8NZ/iMrvi80dtBtRDUUbqf
wvPnLovz/2DD/impOP+1ZKwHHMP74HQCvp4F4v5ZR7A/Lgn9NifC1yEFfXdm
4PUWF4R+ZcJhPFeTGj4HsZZbN3x+LsX86v4i8fz/Meb9vWro3DMqHpA6SLTW
vIaQSt/txetOrWuF84eHkU/2iIM/IZ1U6tuRfGplCPzbYmuE92fyDPi87En1
hwNJtZLlqFuGuczhstBv/n4FcmVBMaliOol1VGfM4TKN1HDXHwj7j1wC387V
pJpmwfrXE6jzcX/MYbdM6Nd2dUK+zfQm1TsNRp36fFK1VR2eawOFftVdwr6Z
LVDnjgnr19zQP2INqXG+TJifRsQtyhXF5yapMddAzvREnitVmEe1DRL7nylC
7lYXIDc35SLvTNewPvYIdRIHi/O7Ihv7OiKnlS6ZWG/LwDx+5cjh6Q+fkN8W
7LNp2N8lFf090rFOPg3fkMtCv3ntUeyvgqqpSejXGWtpZYPf65I4vxcdxn7X
Bv96+KXWKThPRYP/BbHfHHAIny/HUUe1TcA8s5JRd/YZ+O6J/QzDMI3iNzRL
Q78=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 122->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
HOicC4R6hmGYDvEnAqBZuw==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 123->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1: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
aBzGetxZ9JFcKX7/MAzDdIV/AAcpUR4=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 124->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl3tQFWUYxtf7hS6aooaii4qmqKjNKOaFxUsqlU54KS2ZHQYvKArikISj
bsqICROUijGJLEViaIJHBlA5sqBmIoGYBFrmKowyXFJExUIheZ/zR398MYV/
5DTvb+bMM9/u97zf+50537N7nHwCvJa2kyRJfvLp9uTTWWIYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEY5hlH6WlkuT9Rda61vFnNR2633FtRR3V590yz
T/sourxVfs/Po5p9irtLdmv8DMO0wLhXKul8Ddl1k/TF9N/ovM+fdYfG2Y9I
Nb9f7gjPX0RkWfN1Y+2GkmbVLZM+pdxY5HmD6vQYV0vjlZ1qWzq/ms/shGya
9zCa6jknBzWPpV1TrlIfG4PE6/8NRtykBFo3OaGU+rDPqGxNfhiPa/bQ+gdf
rn6q/KnuqnJ+Mc8cTsvoXBrD15Bq5+vpnMnt4mksH7lCak4ZcVf0+9V7Tafz
L7fPIp+ZYI/zntKGVPHeiXH698Lzr4/xPkv3Ax/k0v2Z0Veo3sOrWN/DB7pm
n9BvFjVFUN+dHcNoHyN3pFK9pAOUY9JqFft7e2GL+aO4Jmykfick0nmXLs6N
p/0YvZCDNYUt5o95yW4V3T8hr6cc2xQ3n8ZfzYNv675/lF+qZXPIX+cZ8sbj
tL/oe2Zr8kMpqN9A/iEeq58mf4yotus4v/6HfPIdnWtj+xFS5VAhzvk1E+o/
qo5+P/4r6oTnz8Oe5uknQ6B7fEnVPodQb/oFaEcnoV8pqsf5fLMY+tgP/vfm
kWpNH5CafXOE+WMM7IHn8mm8Z+ir5iN3BqOeGYkc0fJnivMrfSe9p+hfj6+h
9QP7YP7mHsitKB31LC8J/VryGyXImRmUg+aEtciLabV4b2rvCn9Vhji/EvMu
0LqDFbw/XTmM95S6L+Df74jvxWGw0K8Zvqepz7v78mmdzs5URzslV1E/+49S
HcO/Tpg/SmwW/a/S3UIpf+XIUZTH2i2nYtJBlfS9KK6WCuH6h6+l07yYc5RT
0u4og+bvdKb/fcZexyK6n+5TKFzfdSLltZQ9Io368OtmofkuE6muEhBLfalD
Z+cI8yegO/mkOb6khpv/SfI3JFAf2lh7+h9rjNlmFf5++o2k9fWBi2lddcEj
2ofRuyfNN/p509js2mQR+eWYAjwnfrIeJY24nUzrzglJoe8zLCWT9hFWkPws
5qcW3O0+7bOtPanZVH6P+vddhHFEPKnaP+e+qH/V8Vear791llR6J4nUuNUA
XTCPfFpcuNAvH2igXNCTgjDf531S82AE6qUWksrt6u4J128sQD4tbKzD720Y
5nmOR192S0mVtnFCv/bCWOTSknBS082KsRd85tAlpGq2KvQbIX1pvhLrQKqm
BEKdDVzPr0DdjF5CvxLeiHx7rh5qfQHrx87A97LeCzrhQ2F+yjsqkJedbiDH
z19F3l5/CE3sgD4c7MX561IK33DkvtYhHzk+t4hUPoHngFJVJs7fYMxXS86i
TtUpjFefg3rZ6nlcF+dvfzx/5C0nbfu3og9v1JFCUdfsUir0K5E239TjuN8m
E/1+iTq6HepLqZfE+e17DNd/T8e6F9LgX2mrl3gadYLEfmU51jNGQPVgqGSH
vqSP0b/8zWWh/79Ged21nvpLa3xA/ZVXk5oWma4bKzJJlR+L64X7Dw2i+Vpq
Nqm6IwZ+lz9IlbwF8C3bLfTrroHwnRkHX/UwjOvC4V9VjPGUAeL1HUMoV+SO
XeDf+i3GDfbYT956UmO09YHw+y+NQc4dgMqTQ0n1yYdIlQGz4A/2E/rlUa/B
t325LT89oQFlyLtS9KUWLRb7+3VEvmZ0h2/qAIwzx2NcgNw1VtwW5qdUV4mc
2lSBnLpTi/x2bI++zqnIX8tecX43/Ix8nFVi8xeg3hY8B/R4B/gb3MX5nfYD
8nnbeeS3G54Dpk3lsBpcz3EQ+s3cXKz7GVQrOwH/8znwV+bh+pLL4vzfnYX6
TjbNzcC8i6gjDcrH/vYXC/36puOoP9qK+6/CL9+FX5tm868Tr6/1Tse6a46h
76Q0+Dejrr4dfvWm2M8wDPOv+BOO9Ek7
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 125->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztlwtMVnUYxj9FwX1YXlJZEnGwHFqblimEphwrp5HgJTIdNg8oypwOMcUU
iROK5oVJmsvLnJ8XuoErAT/uchSRCAVSUbm5I4hyGyAXRQOW3/tga+1fU2rL
ufe3sWfnnPd5/+85H//nfJ+TX+BsfyuDwSA9+Ov/4K+PgWEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmGYJxzl7dhE9wdqcjtbYVFprfGm+2P49bD6MEu9
9sGiTIuqM4oqHsf/EKn13e8yLP0aryV2x/8QbbGTz7/xM8zTiOJrVU37dNmQ
StKSDfUWlSPkRtr3nb1u0/HhvEbR/pHntpWTr2zOZbru4TvTonraFTov7d1C
Pn2w9e1H2n/elZup34X3t1v2vbLUq8xyrNyKEK7/d+hWO76mdQP7XqX8yX25
pjv7X35h0Bd0/0PTarvlXxyyknxtb0zK6Ib/D+YtD+H8Yv5rTOYg2pf63Hmk
0monUm3jcVIlr53UlDOuSbj/Tbn0XlfaDMiJT11J1bhQ7PfaI+jrlyTe/9qJ
s3TdK/409bG9XUzrdyxDn7OBWN+4ROg3zb0Q/ufzitH9GPl3VlGOaXv74T7W
jH2k/NHaRwbTegFW++m5ZE9A7twr/uf8GXztY8v+NvVbtcei2s0foywqLfUi
n2x2e6T8kvuuX0J9qvNDaf7YjCTyrSjTu7P/tYR6hea5U7Lw3+SPWrJvCufP
04diF0/7Wp51FPvbL5NUdTpEavptTDNpsHez8POvHIZ6/93wZ3ngeOAGUu2Z
nvAPcxT6pdLRqB/RjP2+dTXm2bYA64ccJJXiLgvzR8suqKPr4VHIrXW+6GN0
Rn3zTPiXzBf69V0TK7DPBuB7T49pyJuLAejTnoj8WJorzA85uu4KnZ/e8wbV
jXJGXfMV2u+mjhXoM3Gz0K8OdfkVz30qfU9RTk+j7xmS837yq95mUm3IZKFf
15vPUF141HnSbwoxT0JELXIzFvkTai/0SyddNMq5MFv6nSbXpGfTcU1yIc1x
aRGey3MfVQuf/85g+l2mD1iVSrqimvppLx7OQt99dH/aPacC4fOLMP9EdWUp
CbRe/wNx9Bycys10P9ZGmksJNJ8S+tPCyWfIzYfm3ThJ9QeqMMeeI2mkb404
KXx/ZIfR+urKUlpXTvfBfRSsTyf1vJdC+u3heKH/ToCJ/NPtsP6C3THUJ9aO
+mpjRyVjjlMxT2J+qocGt9L9OQwilaNyW+j/psYF520icf6T5Fbh8/8+kurV
5XEt+P+PacHneJpUHjWBfFrMOqFf29KGXPCLgi93Dtbv3NKC5x+L45yGFuHz
e6mS/JpvJ6me1I682f461r8eDHU2C/2yNJrqlV8Ww381BMdVmEu+Pw5zFUcL
/bqzA9bbagOtkOH/YRP0eNc8r7wj9Gv+nci/VxuRT+8ZqV7NXA6f6SD6WN0S
5qfuWAffhJKuvC5G3sX3xvzBrtBlkUK/7H4VPs98+ArPNWE/nkd+tzZgvl0O
Yn8Y6gw9suBrwvtDKz+DHA9HP5ObLsxfQ69srLsuA3WRp+DfhD7StRwcZxUJ
/aZW+PRjKfA/D5U6cV4xYi5t50WhX9qEeu1IIub1NkMbcN5wGX5p5CXx91+f
ZPjHQKWOJKg71lcNmF8+Viy+//8ZRXrtLj4fa1LNvg+pPHAiqTIvCfpZ1V3h
86/acYd8M9NIDSXRpKaFEtVL4+ejr2uk0K/br0G9y5uk6rbx6DN8NzSmnlRv
dRT6tambKVckT1uqk3JykTcBY+B3CEL/WV3z/QXZJpXqlespyLtZ0ci7yefQ
tx3zyDemCv269STkWmkQNMgf/b60Qn25PakS7iH0Szd7Ub1a2g9+VzvoFDfc
RyfuT435WZifSkEd8jGpFvm7sRp51W7E/NEfQlMVcf42FSO/k4uQT6sK4O+4
jn6hvbG+rb3Qb2jIR05/jveG5pMFnZ2D88925bZNmzi/PfCe0I9mdl1PR16f
09DHtat/QKk4f2+lYf4ZUGVQEuo80Udzzka/4YXi9e1TcL0hFX2+SkRdb/hN
TXlYf3ax0C85ot7khT7S/QQcn8CxOvk8+q8tEq/PMAzzOPwOpP1X2g==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 126->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl31UT3ccx68YztDkNM1TrhjH06G2hVA3eS7mIYwd7WdYnkrMnEpyZZ7G
jjw/nPAT1kpz9EAPSjdqaptQpKK5pVNTVkgSdTZ93m3/7DtHtnPm7Hxe53Te
53vv5/39fr7X7/u+V7dPl06Z31SSJPn5X9vnfy0lhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmFec+SFX0Y71OuE8MJ61V2eFTm8wjzqoh5avc8Q4lv4
Kv4/uTYk9B/5GYb5C6pfZQmdK3dXOt+aVPNrvSrL7O7T+betJNUqb94XnT/j
6n3Ih35O16k+OnZn0nM1lAUW0Lj1Bfh2NnnwovNrWLDQRzh/p5w8uh7kJVz/
7zAW562k9VdL2dTfxOV3G+NXh4xxpX1P3r6G9jP7u9LG+DXzuMmUe/a7l9f7
9VMdtic1wv8H8qAHk6j/4B1TOP+Yfxs11Y3OpXGkJ86n/VJS1TUWWpaL6+W9
H4p+f3LdTnwXDDhK51Mtzqd6JcgD/uK10J57heff2LEonc7Jwivfk68u7xaN
t6Wh/txMUsMsd6FfMw3ZQte1AYdonav76LtF6uNTQfO16Es+fZPNC/NH6mIR
SHXTAjdRboRvNVJ9xWbalzGx9qXyR7dfc4jyosXAPfWq3PNHfobsf7n8KrEI
oLxIeLKe9lNTGkf7CMjTG3X+M7tuJF9595X182kFbgGvkj+GnhV+nDv/X9TY
KDrX6qirpMaqHIxnRpMqXQZV0u/37oRK4flv0uMhvhu2kMpRi6Dt15IaAszI
p5/uLPTrdZbIlY5WyKFWs9BH3jRcj/fGfDbBwvxRpn9F3yt6n5N0vgzvHkIO
mbZDX7MHwy/7C/2S26075DePLCefz2Kc0/6OyA3LSOTGtUJhfhjMluWQb89+
ykFjKL5z1NtNUW8yEuOsPUK/svXgVfJZ19I88vTSMqoLsEduaLbwt5kq9Mvn
lqWS32lqBvUb4IZ+blZinpGDkF8pTkK/Ov4A/f9M6vfFBfr3Np2aRuuO7k3f
TdrEZDyXefnC7yfj+qaxeD9YJVBdrmMy+c70ojyXdr2dSeOPtl4RPv8suwjy
FQ0/TfdX942i+t0bYkjnLKG+9Dc7nheuP+Y25b1xhSP5tRL3JKpvM5D6kA6P
T6TnGpZ7Trj/SYWn6HcTPJvWlYcnn6V5PMrIp5lsw9h9bLTw/ZN9z4jf2wq6
r3mODqfneNCP5tVNIii/pbCY8NcxR9VmBY9o34fNq6jvxBQaq7oNjeUr60j1
3Pgq4fNLW0f1ev/jUNsTpMa9mEf2dyGfsfvnQr+8wJzqlKc/od7JAeNPIjAO
S0c/79uJ17euolyRlVrk1NgnpIan7Rv2FUhq8A57JFy/nS3VG/fNR07J/phv
oyX8cePgT1oj9i9Brsn5dlDLVci5CZHop/AtPJf0OUK/1M4EdQdqkHtR76GP
veOgM7wxr70uzE/lRilyzqEY/q9vY7zyDcybP4BU+fKI0K/uQt7LLpnw+V9G
TsbewPhOBd4HRxzE+d0yA/ctLsLncQF9mP2A60+zoJ53hPlrXAqf4qihj/JE
vDfCkrF+fBrup2aL839/Eu4PS4DfOR5jC1zXHqaS6kevC/2G0Ib6yBj07XEG
9a6YT7mcgv43Z4nX7w6/XhOHvltDpT4NffVueA6r8sTvn/8YbcfgaurrYAtS
Q0HzauTuKFLFMxRqVVAt7P/4lsd0PSiBVH52klT3MaV6tf0MzJuwQehXsldQ
vZriQqpFOUN/OYbrpySs/04rod8QdAy5NbqW1NChGLk1zAZ9+c0lVb49/1jY
/2e34DuRDbXKJ9W61SG3LnVDPyM+FvrljJ5Up47wQh+d52Eel1z0YdsJfl9n
oV/Z3hy+LmbIt3HWyFvf6Vj/hA/GPmeF+ad4lSEfh95F7ra+h7z5sC32YTcU
86TPFedneD5yLicHuXs4Azk1ohDz/dYM6/9sIfaHXIa/WUN+21/E+kOR23rU
A1LNu1qYf7oX3hPGumT48xPhS9Owr16XkL/BOWL/N6jXnKHKB3HQgQ3zJKIv
bZHYr7qfxfolCVgvPha+xUnQzB/h75on9BvCYuDfhnnUotPo1xdjOaLh+Vjn
ivOfYRimMfwOZB5ScQ==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 127->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztlwtQVGUYhg/BWmqljtkoqR28NI6XZr2MaKQeNQXvBkopamuaEg7e0DQm
5BCog06J0shYXo6ilnGRSvECsgeWMEhXuYWYyKroeEEEgUUKNPd7aaaZ/hwv
M+U03zPDvPOf87/f951d/nd33d5f5P2BsyRJ8v2/1vf/npMYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhnnKkdXfk4bdV9uyXRccqizPvTTsMero7ZYe
dvhMpydfeBz/n9iqpu18Ej/DMH/HNDrliuNcyd2tFx2qJabcpPVEv0o6/xFO
VXT+RxRWis6f6jOAfPqRlwtp36XccLNjvSSJrktFR8kn92xT9aDzq4+6Gkj7
zgdvobx4qSGa9veKKia9PUjY/59Qg/P8HHPI8p4Cqrd1y9Unyo/evtcexS/f
2ONlfpJ+DPMvYMudjvNdHkgqOy0gVTfuJ7VdPktqWtT1tuj/Xysde5l8zh7w
m+fSOdUnrMe5nxRF17Ws48Lzr4a8nkP9Y347Tn0ORpeQX/HHeU/fS6qERQn9
sm8flebcN2o77as78wOt9VW3qF5JP/iyhjw4f/wOrKV5qyOXO86tYhgTRX7X
dMwRt/qB+WNa03ch+T8b/DXlTvfF39L5L8D8WlDIQ+WX0pD1xV9zQ267gJ5H
b2z1SN+fTKlBHyI/kzdRHp9r8/Hj5JFSXD+D+kc1hvP3r/8fekEinWvTqnRS
1epWTe9zyllaK4UDaK0uHVMtPP9B7WifrU0c8qFxAeo5WVFvRD7u7+0o9Etz
PHC/6DrOp/ETWutdgknlWVtJtaFZ4vyxD6ogf7IPcibXgLya3Yg8mmfEHLs2
i/0R4WX0nFEhlBfqKju+97w1Grm4zIq6w9sL/bK9HX0/MQ1UKAf1iIPkV5dM
RY6eTsQ8YVZh/mixLfPo/rw7VEedFlNOa08j5o8LQB3XaKHfZHgxi+Y1jj9F
GjuS6igLN1AdLbMrniNkvzB/dJc7OvnuNc+k+0HPZNNzPH/1F1rfmEevi+Qb
dl3kt3XaTb/v1NnnU0lDk9Opn38azaW2t9Dzyftdc4XPPy7xO9rnlH+QtFCn
vFMKPKmuknDRQtd3fp4hfP1PeJNPvp5AqkeWm0nfWUFzqLvLjtF9Q6hZ+Pp1
cUf/iDLqK93qgefwm5VGa1d3WivNag8I50+NoN+lekUB3de0+nh6/9aX0u9n
qXP/o1SveULC05if+vBrNTTv3A61NGf8FVrrX86gtbIrhlT3SakVvn5rNtJ+
qeUhUrk+Dv73Kkm1kaZa/F8uEftnVFMu2OZORZ0NCtQnGnPFZ0FfMAj98rFS
5ErvzaR6s1ukyvf9MIfbctRbvaNG+P5Nc6f9mnEi/BmR0I4d4I+ZTKqEhQr9
+s7OtF/O74I5+i9D/20WUtO6GuiJrkK/3PsecsW3EppYjzzMljFH0STkb1W8
MD+V7eXI6dLz8McWIy8DM5GjblXI0TNDhH7bFOyXW+UhJz1z4JuD/NZX/oh5
4p3F/Vecgq91NnK/OgPrbPRXuh3H+s1Scf5GIteVu2m4H2CGGizQV1BXSy8S
+tUpOuobjqC//2HofNRTZ2N+KThf3D89BddLD0H9k6EZTde9MJ8aUCj2v4F9
6ttH8To9C5VXYC69rOl18SwW+v9z3D3qaH5vF1L5RIWd5uzjRmu9RVIdPkdy
6oT/v3sX0X55pYVUmRALjblEqmePg6/bOqFfGpSG/TOHQNfNhK/nIVKbVyOu
W3oI/erG7ZQL8iRXzO11E+tiGXN9tZJU3aTbhee34Vfkm1ceqWYqQQ5+ZEed
xZ0wz+AAoV99dShycmwQdORa+AJ/IjXt64f+nbzF/d2acm1gN2iLvvCdDcdc
N9ajbmiNMP+kpHLkbChUqXDGc6xtjv3+I7DO2CD2e5aQz+ZdjPx2OUWqNtQi
/066YJ4ORqFfGXMauZZ0EnWGI6/lHTmYa3oF5vLoJfZbM3F/twX9XjsG/7sZ
qHcX9dUJ58T5HWdGn2/gky4cRr2ZWCsFP6Pu+EKh3zY/Ff4Q1NHDj6Dfp2mY
w7fp+dqK+9u2NfWzp+BzyojPQdPgpv6SFevYM0I/wzDMI/EHWpdJXw==
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 128->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl3tMVnUYx0/mbU4n6DA00ENmiNlmoUVkcMxEpA0riJRpHElEU/K6UDE9
4AXMS5AoShdOOpbMWeAEuYic1xAFr4gKvb22g4ghYkIgaJnm+3yxufWzhf/k
2vPZ2He/c37f53neA7/ve3ALn/t2xOOSJMl3fxzu/nSXGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZh
GIZhGIZhGIZhGIZhGIZhGIZ5xDG6T8zwvavqknyT9MPCC74d8KuHVkXb92uX
NuaQ3zrF7Ij/Hnpd6FK7z3irYMvD+BmGeTDym2tr6VxF3qymc+rtd9Wu8oZX
G+2qON0hlRpONIrOn5Kzg3zS6i0V99/X5szDdfep5NNzbgn9f82R4B9eZN9n
vfwpnfe0b2JJL6pVlCN1/f/R/7d6FjneXk+yRdBc5gc96h4mP0zXMe/T86jL
fig/wzzSvBfSROesIopU6TaDVA1LJ5Xn26AuA38V/f2brUMu0f2xHrTPXL6U
VHd7A3WPeULP5TWJ/Maam2Xkv2g5Qr7Q4vO0dl6DOo7+qLt6idAv/Tx7Ae0P
fW0rzZ3bmEn9XHdeo/3TUyk3zE4+Yv+9OYJ7zLz/vu57NYHWrbOQO4dX/WP+
GMO9Q6j/mbBd9txR00eTapZ88hnj1H+VX/JU7612n5kcn0y+GTXZ5BviXt2h
/DueNoP867ISizrgexDG/NdjOf/+f+hmFp1reXIGqdLQpZm+b2ustNbnedFa
PfpOs/D8r3OC/zsv6Gd9obeDSdUBR7E+6Sr0G+nP0X3jBnyS+gfOe+YdnNeF
QfCHHRfmj7I5gM65eiUQ+bXtc1Lt2YPIjxhnzJGWKvSrw7zp/UeZgPcLfaEv
8q5uLnIrswbrDc5if3zuD9Tv6g2qo9nqcM4dM3HuK7/CPIPLhfmjGItOU/1h
pVb63G596P1L6hVEfrXCC/kV7S70y72TS2j+o31PUZ/0RppHd3GiOqpZC3/a
FWH+aB6TLTTnR9HFpK9sLCVfhFpJ+0MvIEd//6VeeP5XeudR3x/XFVKfst+o
nt7P9zDtDxhM71/y9KRy4fOLGrqHrkeE0f+JSmHgXppj++VcqqsH0Fxyy/WD
wv6Lc7Lxex9Efq2Tp0HztxbQHMqc5QdII50M4fN3+CKL6p99l/qq/dfvp/7V
pfD1K6e1nLQ1W/j8ertsp/slnnRfa8vYTf6ZY6mutjgnn9bR6d8+ivmpJd5q
ofm+7H+d5i2sp7UaF0prPWAbqeSz97rw/Pa8AH/eblKt2YK11UZqVo4kn/Ji
jNBvzm6hXDB/KqP9yoIo6O6dpHpWCebp0k/srz9PfuWTrth3cDR8uwLaP9ci
XA/a3yKcf9ko5JIlhlSvKCCVp4zAHIGB8EdmCv2KMRD9rzkgNx0ikZd+tdCx
nphnpSr063seQ79NN5Bzfd0xx4AUUiOuFPUd6oX5KU+rJ5+5rBp57WMi76zP
oE5PFf7YcKFfn2Nr73savvGnSLVpZ5HLm5/AfD6dxfldchI+7yPoH/w9cnLV
CVx/+Qyuv1QrzE8pHz7tkoF9kw5AC4oxl1sp5lhrFfqN+fDJywpQpyofz8ML
16Xcw6jnf1bo1+P2w1+0D/smQo3w9uuJJViPqBC//6Zhn9KU3/59iTnkFAuu
R5fhemfx/P81ZiffNpqzobmVfl9PNUH9XOm6MbKYVMspaxPmd/IK2q8k7SU1
Y1NINa0GumJqG/4uE4R+3WJgX7ML/EcmYL1pF6mUWo95NEdxf49EygVjfFfa
p98ykWPVjrQ2YpLgVy2twv76aeRK/HrkVM9M5N61c1i/4IzP1yNc6JfOBNI+
ddQk9O2VinpdSjBXsis+zxh/cX9bN/if7wpf1dNYR4XAn5ZOatbahPmnxjQg
d49cgc66g7yp6I55xgxCnXEp4vzccR751gd5rS0sJ5UTKpFXGbeRgwOeFPqV
j08i58uPoW/0IWhcKbQOuS5XDBP6JT/kuxFSjH1fF2KeTfgeUUOOo76HTZy/
a4rgC4JPW5QHzT4A/znMZUiVQr/WDfvM6Pa+1blYD0Fdffgp+BqrxP175WOf
P/xS8D6sJayNxScwz1Cr0M8wDNMh/gRRSE7Z
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}], 129->
                Graphics3DBox[
                 TagBox[Raster3DBox[CompressedData["
1:eJztl2tQF1UYxtdEAZu8C0lZi5QO4pQXyAsy/PMGWXnB1EDJHQQdFcIrOBPK
FtaYmoYjaI7gYmqmguINEZAVJC8IcklFMFgZFQYkQBCUSvP/PkxfOllSM/nh
/c0wz+zZ87znPQvn2cXeL9groL0kSfLjn66Pf6wkhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEY
hmEYhmEYhmEYhmEYhmGecUyBiuZu1tfsy8wqH9h6w70tdcrC480+5VWv0rb4
/8C15+J/5WcY5k+oQZNums+VXjGLzrdpj3uNWTWL4Do6t7271tP4/Lw60fkz
7nxPPnVSTSHNGx6yiealPVdOatuX/HKoVf2Tzq8x1XV3utkfmLKe6iz18zer
MeXFIvJv7iFc/+8w5pTl0z4GRla0KT9WWPYx9yWvGV/5NH5lty31b6qMnEPP
1y5uZZv6Dwsiv/aB5yzOP+a/xjTal86lNnYBqZFZS+dMt03Gee17klQ5/cpd
4fkfsfc2nf9+28in7tlBKlck4bzeW4r6ocuE519rsMih+TcGZZM259P3hjbB
gearnWdj/dkBQr8aUzif5sdosTS/T5fjdO4i59L6pm86wO/TX5w/5fHBtN/E
Yctp3pAT9N2jjo2LpnGvaDwPffQT80eeGbDfnBPSm3GkRqMab1Y9dyL5jGiX
f5RfxtB2ceT7ZXYE9XN0yRHaX9Md42nOvx5qE0F5Wjcm1qyK94bVbckPOfGQ
Lz1H99t709vgZ55tTC3xdK6VoGxSred9UqN9MjTRoYHGq95uEP39mD4dSvPU
juWk8rQg1HHYh+u5Rbj2txH7DTe6ry+5gvPeNP0u3v+h8NuuQP0OccL8kZd1
QO6MnUl+fX4UzqtUge+WnqX4/mgXLPTrmfdv0f53fYw6G9bju+fYHmhJHPIr
8JEwP5Qt5cVUv9iHvi/0GRuxfixyS/OORl92x4V+udAoIL/xxnXS96/Q95ei
eSK3hvjBt6hWmB9agMNZ2mfSujzqPyGD+jGs0qmO6pODPA6qEvpN1mmnqb+c
bllUp5vPBVo31KkI//8tRn6Ne/eOsH+7q8nkj3/nFOmA8Axar88t6kv6qoi+
C/WWkAJh/ry38zD1OaKKcls9U3WU1puWQHVlLf0MjVsuyBSu7+pAPn3UZlJp
XYhOvkpr6kO77JhO+wkI1oXPz82S1tf7XaV11cCmVLrOHk0+bdeKNLyXPI4L
3z/jT+6k5+QddYz0S/8EUud9ibQPa7cUquu9/eAz+f0Wcb+R9mvd+x7ts985
XA93pmvDaTmpqVPSPeH5OZBA802hkaTaDZXU2BGNOtUe5FPCFwr92gZb+BaE
wVfoRSrZrYTaf4dx/8Hi9bt3pPtKgiP68BwG35q34BscC0082SjyG+EulEsm
l1WkysJqUnVJC6mRin709puEfumBjHkDTKRy9zDkZd3XqLfxV9S/Eij2336I
XPqwAdruJfgtLOB3mYz6Uq4wP5Vt1cjXTshfY14p6oQ9QG76dkZf13aL83de
CeYrBcjbC7nw9cjHe2HUz8jjwyOFfnlgHuaXnsX8g1m49oTKFedxnVcmzF+l
+BzuF+jYR690rDfqB4yPxH355RJx/k88jXH/VMyrx3tLs0c9Uy36kFoui/0T
0jAvotW3LAnXNhhXSlr7mFMo7r8ltXW9lNbfI1Qpwvp6zAWMh1wT+v9vtG5j
mqnPvJom6nvK83Rt7OtPKmdkkkqBxc3C57cjinxyzSFStcsJUi3Piuarlzyg
yTFi//YMrHvGGRo6DfWy95NKn9WS6q6/NQnP70dRlAv6jHrk19bryCtHJ/Qx
+XP4j2UI/VLAVeTTw0josVOolyXRfNO4gehn/Frx+hfdkJOL/JBPlshL44Vy
1LPsi311mif0y1GWNE9b2Ava2QX+I2HQTVnYT1GVMP+UH2uQf4+qkVO3mqFH
kOfqlolQ99Xi/NR/Qi7ZFEN35cG/CvVMKT3h/8LpL/I7H+uvzYUv4izqLD6P
XPevx3ijJPQbe7OQz3GZ0MpU+F7PwPrbc+CfXizMT2W9jvEZ6ZjXfAI69RSp
7HgR9ZyvivN3Sxru38R8xSMF+klr3UGXSNVE8fpaA94rRjn6Vr5NQr1SXMvl
eJ6y7zXx+gzDME/D74v9WQk=
                   "], {{0, 16, 32}, {8, 0, 0}}, {0., 1.},
                   ColorFunction->"GrayLevelDefaultColorFunction",
                   Method->{"FastRendering" -> True}],
                  BoxForm`ImageTag[
                  "Real", ColorSpace -> Automatic, Interleaving -> None],
                  Selectable->False],
                 Axes->True,
                 AxesLabel->{"x", "y", "z"},
                 AxesStyle->{},
                 Background->None,
                 BaseStyle->"Image3DGraphics3D",
                 BoxRatios->Automatic,
                 Boxed->True,
                 ImageSizeRaw->8,
                 PlotRange->{{0, 8}, {0, 16}, {0, 32}},
                 ViewPoint->{-2.4, 1.1, 1.}]}, Dynamic[$CellContext`i12$$],
                Alignment->Automatic,
                ImageSize->All],
               Identity,
               Editable->True,
               Selectable->True],
              ImageMargins->10],
             Deployed->False,
             StripOnInput->False,
             ScriptLevel->0,
             GraphicsBoxOptions->{PreserveImageOptions->True},
             Graphics3DBoxOptions->{PreserveImageOptions->True}],
            Identity,
            Editable->False,
            Selectable->False],
           Alignment->{Left, Center},
           Background->GrayLevel[1],
           Frame->1,
           FrameStyle->GrayLevel[0, 0.2],
           ItemSize->Automatic,
           StripOnInput->False]}
        },
        AutoDelete->False,
        GridBoxAlignment->{
         "Columns" -> {{Left}}, "ColumnsIndexed" -> {}, "Rows" -> {{Top}}, 
          "RowsIndexed" -> {}},
        GridBoxDividers->{
         "Columns" -> {{False}}, "ColumnsIndexed" -> {}, "Rows" -> {{False}}, 
          "RowsIndexed" -> {}},
        GridBoxItemSize->{"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}},
        GridBoxSpacings->{"Columns" -> {
            Offset[0.7], {
             Offset[0.5599999999999999]}, 
            Offset[0.7]}, "ColumnsIndexed" -> {}, "Rows" -> {
            Offset[0.4], {
             Offset[0.8]}, 
            Offset[0.4]}, "RowsIndexed" -> {}}], If[
        CurrentValue["SelectionOver"], 
        Manipulate`Dump`ReadControllerState[
         Map[Manipulate`Dump`updateOneVar[#, 
           CurrentValue["PreviousFormatTime"], 
           CurrentValue["CurrentFormatTime"]]& , {
           
           Manipulate`Dump`controllerLink[{$CellContext`i12$$, \
$CellContext`i12$2121340$$}, "X1", 
            If["DefaultAbsolute", True, "JB1"], False, {1, 129, 1}, 129, 
            1.]}], 
         CurrentValue[{
          "ControllerData", {
           "Gamepad", "Joystick", "Multi-Axis Controller"}}], {}]],
       ImageSizeCache->{311., {241.75, 248.75}}],
      DefaultBaseStyle->{},
      FrameMargins->{{5, 5}, {5, 5}}],
     BaselinePosition->Automatic,
     ImageMargins->0],
    Deinitialization:>None,
    DynamicModuleValues:>{},
    SynchronousInitialization->True,
    UnsavedVariables:>{Typeset`initDone$$},
    UntrackedVariables:>{Typeset`size$$}], "ListAnimate",
   Deployed->True,
   StripOnInput->False],
  Manipulate`InterpretManipulate[1]]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"velocity", " ", "u"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{"Animate", "[", 
   RowBox[{
    RowBox[{"VisualizeVelocityField", "[", 
     RowBox[{
      RowBox[{
       SubscriptBox["vel", "evolv"], "\[LeftDoubleBracket]", 
       RowBox[{"t", "+", "1"}], "\[RightDoubleBracket]"}], ",", 
      RowBox[{"\"\<t: \>\"", "<>", 
       RowBox[{"ToString", "[", "t", "]"}]}]}], "]"}], ",", 
    RowBox[{"{", 
     RowBox[{"t", ",", "0", ",", 
      SubscriptBox["t", "max"], ",", "1"}], "}"}], ",", 
    RowBox[{"AnimationRunning", "\[Rule]", "False"}]}], "]"}]}]], "Input"],

Cell[BoxData[
 TagBox[
  StyleBox[
   DynamicModuleBox[{$CellContext`t$$ = 13, Typeset`show$$ = True, 
    Typeset`bookmarkList$$ = {}, Typeset`bookmarkMode$$ = "Menu", 
    Typeset`animator$$, Typeset`animvar$$ = 1, Typeset`name$$ = 
    "\"untitled\"", Typeset`specs$$ = {{
      Hold[$CellContext`t$$], 0, 128, 1}}, Typeset`size$$ = {
    360., {193., 198.}}, Typeset`update$$ = 0, Typeset`initDone$$, 
    Typeset`skipInitDone$$ = True, $CellContext`t$2121618$$ = 0}, 
    DynamicBox[Manipulate`ManipulateBoxes[
     1, StandardForm, "Variables" :> {$CellContext`t$$ = 0}, 
      "ControllerVariables" :> {
        Hold[$CellContext`t$$, $CellContext`t$2121618$$, 0]}, 
      "OtherVariables" :> {
       Typeset`show$$, Typeset`bookmarkList$$, Typeset`bookmarkMode$$, 
        Typeset`animator$$, Typeset`animvar$$, Typeset`name$$, 
        Typeset`specs$$, Typeset`size$$, Typeset`update$$, Typeset`initDone$$,
         Typeset`skipInitDone$$}, 
      "Body" :> $CellContext`VisualizeVelocityField[
        Part[
         Subscript[$CellContext`vel, $CellContext`evolv], $CellContext`t$$ + 
         1], 
        StringJoin["t: ", 
         ToString[$CellContext`t$$]]], 
      "Specifications" :> {{$CellContext`t$$, 0, 128, 1, AnimationRunning -> 
         False, AppearanceElements -> {
          "ProgressSlider", "PlayPauseButton", "FasterSlowerButtons", 
           "DirectionButton"}}}, 
      "Options" :> {
       ControlType -> Animator, AppearanceElements -> None, DefaultBaseStyle -> 
        "Animate", DefaultLabelStyle -> "AnimateLabel", SynchronousUpdating -> 
        True, ShrinkingDelay -> 10.}, "DefaultOptions" :> {}],
     ImageSizeCache->{411., {231., 238.}},
     SingleEvaluation->True],
    Deinitialization:>None,
    DynamicModuleValues:>{},
    SynchronousInitialization->True,
    UnsavedVariables:>{Typeset`initDone$$},
    UntrackedVariables:>{Typeset`size$$}], "Animate",
   Deployed->True,
   StripOnInput->False],
  Manipulate`InterpretManipulate[1]]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"density", " ", "\[Rho]"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{"Animate", "[", 
   RowBox[{
    RowBox[{"Visualize3DTable", "[", 
     RowBox[{
      SubscriptBox["\[Rho]", "evolv"], "\[LeftDoubleBracket]", 
      RowBox[{"t", "+", "1"}], "\[RightDoubleBracket]"}], "]"}], ",", 
    RowBox[{"{", 
     RowBox[{"t", ",", "0", ",", 
      SubscriptBox["t", "max"], ",", "1"}], "}"}], ",", 
    RowBox[{"AnimationRunning", "\[Rule]", "False"}]}], "]"}]}]], "Input"],

Cell[BoxData[
 TagBox[
  StyleBox[
   DynamicModuleBox[{$CellContext`t$$ = 53, Typeset`show$$ = True, 
    Typeset`bookmarkList$$ = {}, Typeset`bookmarkMode$$ = "Menu", 
    Typeset`animator$$, Typeset`animvar$$ = 1, Typeset`name$$ = 
    "\"untitled\"", Typeset`specs$$ = {{
      Hold[$CellContext`t$$], 0, 128, 1}}, Typeset`size$$ = {
    239., {214., 218.}}, Typeset`update$$ = 0, Typeset`initDone$$, 
    Typeset`skipInitDone$$ = True, $CellContext`t$2119663$$ = 0}, 
    DynamicBox[Manipulate`ManipulateBoxes[
     1, StandardForm, "Variables" :> {$CellContext`t$$ = 0}, 
      "ControllerVariables" :> {
        Hold[$CellContext`t$$, $CellContext`t$2119663$$, 0]}, 
      "OtherVariables" :> {
       Typeset`show$$, Typeset`bookmarkList$$, Typeset`bookmarkMode$$, 
        Typeset`animator$$, Typeset`animvar$$, Typeset`name$$, 
        Typeset`specs$$, Typeset`size$$, Typeset`update$$, Typeset`initDone$$,
         Typeset`skipInitDone$$}, "Body" :> $CellContext`Visualize3DTable[
        Part[
         Subscript[$CellContext`\[Rho], $CellContext`evolv], $CellContext`t$$ + 
         1]], "Specifications" :> {{$CellContext`t$$, 0, 128, 1, 
         AnimationRunning -> False, 
         AppearanceElements -> {
          "ProgressSlider", "PlayPauseButton", "FasterSlowerButtons", 
           "DirectionButton"}}}, 
      "Options" :> {
       ControlType -> Animator, AppearanceElements -> None, DefaultBaseStyle -> 
        "Animate", DefaultLabelStyle -> "AnimateLabel", SynchronousUpdating -> 
        True, ShrinkingDelay -> 10.}, "DefaultOptions" :> {}],
     ImageSizeCache->{333., {251., 258.}},
     SingleEvaluation->True],
    Deinitialization:>None,
    DynamicModuleValues:>{},
    SynchronousInitialization->True,
    UnsavedVariables:>{Typeset`initDone$$},
    UntrackedVariables:>{Typeset`size$$}], "Animate",
   Deployed->True,
   StripOnInput->False],
  Manipulate`InterpretManipulate[1]]], "Output"]
}, Open  ]],

Cell[CellGroupData[{

Cell[BoxData[
 RowBox[{
  RowBox[{"(*", " ", 
   RowBox[{"internal", " ", "energy"}], " ", "*)"}], "\[IndentingNewLine]", 
  RowBox[{"Animate", "[", 
   RowBox[{
    RowBox[{"Visualize3DTable", "[", 
     RowBox[{
      SubscriptBox["en", 
       RowBox[{"int", ",", "evolv"}]], "\[LeftDoubleBracket]", 
      RowBox[{"t", "+", "1"}], "\[RightDoubleBracket]"}], "]"}], ",", 
    RowBox[{"{", 
     RowBox[{"t", ",", "0", ",", 
      SubscriptBox["t", "max"], ",", "1"}], "}"}], ",", 
    RowBox[{"AnimationRunning", "\[Rule]", "False"}]}], "]"}]}]], "Input"],

Cell[BoxData[
 TagBox[
  StyleBox[
   DynamicModuleBox[{$CellContext`t$$ = 52, Typeset`show$$ = True, 
    Typeset`bookmarkList$$ = {}, Typeset`bookmarkMode$$ = "Menu", 
    Typeset`animator$$, Typeset`animvar$$ = 1, Typeset`name$$ = 
    "\"untitled\"", Typeset`specs$$ = {{
      Hold[$CellContext`t$$], 0, 128, 1}}, Typeset`size$$ = {
    239., {214., 218.}}, Typeset`update$$ = 0, Typeset`initDone$$, 
    Typeset`skipInitDone$$ = True, $CellContext`t$2121210$$ = 0}, 
    DynamicBox[Manipulate`ManipulateBoxes[
     1, StandardForm, "Variables" :> {$CellContext`t$$ = 0}, 
      "ControllerVariables" :> {
        Hold[$CellContext`t$$, $CellContext`t$2121210$$, 0]}, 
      "OtherVariables" :> {
       Typeset`show$$, Typeset`bookmarkList$$, Typeset`bookmarkMode$$, 
        Typeset`animator$$, Typeset`animvar$$, Typeset`name$$, 
        Typeset`specs$$, Typeset`size$$, Typeset`update$$, Typeset`initDone$$,
         Typeset`skipInitDone$$}, "Body" :> $CellContext`Visualize3DTable[
        Part[
         Subscript[$CellContext`en, $CellContext`int, $CellContext`evolv], \
$CellContext`t$$ + 1]], 
      "Specifications" :> {{$CellContext`t$$, 0, 128, 1, AnimationRunning -> 
         False, AppearanceElements -> {
          "ProgressSlider", "PlayPauseButton", "FasterSlowerButtons", 
           "DirectionButton"}}}, 
      "Options" :> {
       ControlType -> Animator, AppearanceElements -> None, DefaultBaseStyle -> 
        "Animate", DefaultLabelStyle -> "AnimateLabel", SynchronousUpdating -> 
        True, ShrinkingDelay -> 10.}, "DefaultOptions" :> {}],
     ImageSizeCache->{333., {251., 258.}},
     SingleEvaluation->True],
    Deinitialization:>None,
    DynamicModuleValues:>{},
    SynchronousInitialization->True,
    UnsavedVariables:>{Typeset`initDone$$},
    UntrackedVariables:>{Typeset`size$$}], "Animate",
   Deployed->True,
   StripOnInput->False],
  Manipulate`InterpretManipulate[1]]], "Output"]
}, Open  ]]
}, Open  ]],

Cell[CellGroupData[{

Cell["Clean up", "Section"],

Cell[BoxData[
 RowBox[{
  RowBox[{"Uninstall", "[", "lbmLink", "]"}], ";"}]], "Input"]
}, Open  ]]
}, Open  ]]
},
WindowSize->{1167, 1125},
WindowMargins->{{Automatic, 473}, {Automatic, 0}},
ShowSelection->True,
TrackCellChangeTimes->False,
FrontEndVersion->"10.0 for Microsoft Windows (64-bit) (July 1, 2014)",
StyleDefinitions->"Default.nb"
]
(* End of Notebook Content *)

(* Internal cache information *)
(*CellTagsOutline
CellTagsIndex->{
 "Info3628000100-8197276"->{
  Cell[71680, 1565, 239, 4, 40, "Print",
   CellTags->"Info3628000100-8197276"]}
 }
*)
(*CellTagsIndex
CellTagsIndex->{
 {"Info3628000100-8197276", 528060, 10016}
 }
*)
(*NotebookFileOutline
Notebook[{
Cell[CellGroupData[{
Cell[1486, 35, 47, 0, 90, "Title"],
Cell[1536, 37, 926, 19, 258, "Text"],
Cell[2465, 58, 123, 3, 31, "Input"],
Cell[2591, 63, 187, 5, 31, "Input"],
Cell[2781, 70, 411, 12, 52, "Input"],
Cell[CellGroupData[{
Cell[3217, 86, 35, 0, 63, "Section"],
Cell[CellGroupData[{
Cell[3277, 90, 1924, 60, 72, "Input"],
Cell[5204, 152, 29, 0, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[5270, 157, 359, 11, 31, "Input"],
Cell[5632, 170, 188, 5, 210, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[5857, 180, 480, 15, 52, "Input"],
Cell[6340, 197, 29, 0, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[6406, 202, 269, 7, 72, "Input"],
Cell[6678, 211, 83, 2, 31, "Output"],
Cell[6764, 215, 28, 0, 31, "Output"]
}, Open  ]],
Cell[6807, 218, 1191, 36, 67, "Input"],
Cell[8001, 256, 117, 3, 31, "Input"],
Cell[8121, 261, 826, 24, 46, "Input"],
Cell[8950, 287, 1027, 31, 46, "Input"],
Cell[9980, 320, 893, 27, 46, "Input"],
Cell[CellGroupData[{
Cell[10898, 351, 429, 12, 52, "Input"],
Cell[11330, 365, 33, 0, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[11400, 370, 603, 18, 52, "Input"],
Cell[12006, 390, 152, 5, 31, "Output"]
}, Open  ]],
Cell[12173, 398, 641, 20, 112, "Input"],
Cell[12817, 420, 1888, 50, 132, "Input"],
Cell[14708, 472, 665, 17, 72, "Input"],
Cell[15376, 491, 776, 22, 52, "Input"]
}, Open  ]],
Cell[CellGroupData[{
Cell[16189, 518, 44, 0, 63, "Section"],
Cell[16236, 520, 282, 9, 72, "Input"],
Cell[CellGroupData[{
Cell[16543, 533, 4063, 105, 152, "Input"],
Cell[20609, 640, 85, 2, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[20731, 647, 313, 8, 52, "Input"],
Cell[21047, 657, 509, 18, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[21593, 680, 96, 2, 31, "Input"],
Cell[21692, 684, 27943, 441, 447, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[49672, 1130, 1434, 42, 96, "Input"],
Cell[51109, 1174, 96, 2, 31, "Output"]
}, Open  ]],
Cell[51220, 1179, 815, 25, 72, "Input"],
Cell[CellGroupData[{
Cell[52060, 1208, 705, 21, 52, "Input"],
Cell[52768, 1231, 17919, 285, 391, "Output"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[70736, 1522, 33, 0, 63, "Section"],
Cell[70772, 1524, 123, 4, 31, "Input"],
Cell[70898, 1530, 261, 8, 52, "Input"],
Cell[71162, 1540, 94, 3, 31, "Input"],
Cell[CellGroupData[{
Cell[71281, 1547, 267, 8, 52, "Input"],
Cell[71551, 1557, 30, 0, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[71618, 1562, 59, 1, 31, "Input"],
Cell[71680, 1565, 239, 4, 40, "Print",
 CellTags->"Info3628000100-8197276"]
}, Open  ]],
Cell[71934, 1572, 638, 19, 31, "Input"],
Cell[CellGroupData[{
Cell[72597, 1595, 277, 6, 72, "Input"],
Cell[72877, 1603, 97, 2, 31, "Output"],
Cell[72977, 1607, 108, 2, 31, "Output"],
Cell[73088, 1611, 97, 2, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[73222, 1618, 270, 7, 52, "Input"],
Cell[73495, 1627, 644, 12, 52, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[74176, 1644, 1454, 43, 72, "Input"],
Cell[75633, 1689, 6319, 152, 258, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[81989, 1846, 959, 28, 72, "Input"],
Cell[82951, 1876, 97, 2, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[83085, 1883, 427, 12, 72, "Input"],
Cell[83515, 1897, 107, 2, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[83659, 1904, 462, 13, 72, "Input"],
Cell[84124, 1919, 97, 2, 31, "Output"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[84270, 1927, 36, 0, 63, "Section"],
Cell[CellGroupData[{
Cell[84331, 1931, 515, 13, 31, "Input"],
Cell[84849, 1946, 1993, 41, 526, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[86879, 1992, 92, 2, 31, "Input"],
Cell[86974, 1996, 97, 2, 31, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[87108, 2003, 594, 15, 52, "Input"],
Cell[87705, 2020, 432093, 7786, 526, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[519835, 9811, 637, 16, 52, "Input"],
Cell[520475, 9829, 2003, 42, 486, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[522515, 9876, 531, 13, 52, "Input"],
Cell[523049, 9891, 1930, 39, 526, "Output"]
}, Open  ]],
Cell[CellGroupData[{
Cell[525016, 9935, 558, 14, 52, "Input"],
Cell[525577, 9951, 1933, 39, 526, "Output"]
}, Open  ]]
}, Open  ]],
Cell[CellGroupData[{
Cell[527559, 9996, 27, 0, 63, "Section"],
Cell[527589, 9998, 86, 2, 31, "Input"]
}, Open  ]]
}, Open  ]]
}
]
*)

(* End of internal cache information *)

(* NotebookSignature 5wTt83FlsPhvlAKVXYPbA1lV *)
