
target("ppl")
    set_kind("static")
    add_includedirs("$(projectdir)/include")
    add_files("host/*.cpp")
    add_packages("glm")
    if is_plat("linux") then
        add_packages("pthread")
    end