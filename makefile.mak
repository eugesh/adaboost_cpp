base_dir=$(BASE_DIR)
include $(MAKE_FOR_SYSTEM)

##############################################

# ��� ������������ �����
name_exe=abc_class_lib.dll

# ���������� � ��������� �������
source_dir=$(base_dir)/source

# ���������� ��� ��������� ������
obj_dir=$(base_dir)/obj

# ���������� ��� moc-������
moc_dir=$(base_dir)/moc

# ���������� ��� ������������ �����
proc_dir=$(base_dir)/exe

# ���������� � h-������� ������� ����
source_header_form_dir=$(base_dir)/form_h

# ���������� � ui-������� ������� ����
source_form_dir=$(base_dir)/form_ui

# ���������� � ������������� �������
cpp_options_include=$(opt_inc) "$(source_dir)" \
							$(opt_inc) "$(source_header_form_dir)"

# h-����� � ������� ��� ����
#h_file_widget=\
#	$(source_header_form_dir)/ui_viewer.h

# ��������� ����� �� ����������� moc-������
obj_file=\
	$(obj_dir)/adaboost_cumsum_lib.$(ext_obj)
#	$(obj_dir)/main.$(ext_obj) \

# ��������� ����� ���������� �� moc-������						
#obj_moc_file=\
#	$(obj_dir)/cviewer_moc.$(ext_obj) \
#	$(obj_dir)/cscene_moc.$(ext_obj)

all:
	$(MAKE) $(opt_make_f) makefile.mak clean
	$(MAKE) $(opt_make_f) makefile.mak program

clean:
	$(command_del) $(base_dir)$(SD)obj$(SD)*.$(ext_obj)
	$(command_del) $(base_dir)$(SD)moc$(SD)*moc.cpp
	$(command_del) $(obj_file)
	$(command_del) $(obj_moc_file)
	$(command_del) $(proc_dir)/$(name_exe)

# �������� ���� "���������� ����������"
program :  $(proc_dir)/$(name_exe)

# �������� h-������ �� ������ ui-������
#$(source_header_form_dir)/ui_viewer.h : $(source_form_dir)/ui_viewer.ui
#	$(UIC) -o $@ $?

# �������� moc-������
#$(moc_dir)/cviewer_moc.cpp : $(source_dir)/cviewer.h
#	$(MOC) -o $@ $?

#$(moc_dir)/cscene_moc.cpp : $(source_dir)/cscene.h
#	$(MOC) -o $@ $? 

# �������� �����

#$(obj_dir)/main.$(ext_obj) : $(source_dir)/main.cpp
#	$(command_comp)

$(obj_dir)/adaboost_cumsum_lib.$(ext_obj) : $(source_dir)/adaboost_cumsum_lib.cpp
	$(command_comp)

$(source_dir)/adaboost_cumsum_lib.cpp : \
		$(source_dir)/adaboost_cumsum_lib.h
	$(command_touch)

######################################################

#  ����������� �������������� �������
#include $(base_dir)/makefile_lib.mak

######################################################

# �������� ���������� � ������� ����������
$(proc_dir)/$(name_exe) : $(obj_file)
	$(command_linkDLL)  $(obj_file) $(library)


