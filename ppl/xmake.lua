
target("ppl")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files("host/*.cpp")
    add_packages("glm")
    add_cxxflags("-pthread")
    -- if is_plat("linux") then
    --     add_packages("pthread")
    -- end