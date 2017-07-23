//
// Created by oleg on 23.07.17.
//

#include <Types.hpp>
#include <sstream>

Size::Size():
    m_Width(0),
    m_Height(0)
{ }

Problem::Problem():
    m_InputMaps(0),
    m_OutputMaps(0)
{ }

Result::Result():
    m_Repeats(0),
    m_ForwardElapsedUSec(0),
    m_BackwardFilterElapsedUSec(0),
    m_BackwardInputsElapsedUSec(0)
{ }
