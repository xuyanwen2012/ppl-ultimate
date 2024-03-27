target("new-hybrid")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("../../include/**/*")
    add_files("*.cu", "*.cpp")
    add_packages("glm")
    if(is_plat("linux")) then
        add_packages("pthread")
    end
    add_deps("ppl-hybrid")
    add_cugencodes("native")
