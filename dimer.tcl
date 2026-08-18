############################################################
# Loads a pdb file and trajectory file
############################################################
proc load_traj {pdb_file trajectory_file {stride 1}} {
  if { [catch {open $pdb_file r} fid] } {
    puts stderr "File does not exist.\n"
  }

  #Loads in pdb and trajectory files
  set molid [mol new $pdb_file waitfor all]
  mol addfile $trajectory_file step $stride waitfor all

  #Creates box
  color Display Background white
  pbc wrap -center origin -all
  pbc box -color black -center origin

  #mol selection all
  mol delrep 0 $molid

  pbc wrap -center origin -all
  pbc box -color black -center origin
  #mol selection all
  mol delrep 0 $molid

  #Creates the A dimers
  mol addrep $molid
  mol modcolor 0 $molid ColorID 1
  mol modstyle 0 $molid VDW 1.000000 8.000000
  mol modmaterial 0 $molid Opaque
  mol modselect 0 $molid {segname "A.*"}

  #Creates the B dimers
  mol addrep $molid
  mol modcolor 1 $molid ColorID 8
  mol modstyle 1 $molid VDW 1.000000 8.000000
  mol modmaterial 1 $molid Opaque
  mol modselect 1 $molid {segname "B.*"}

  #Creates the C dimers
  mol addrep $molid
  mol modcolor 2 $molid ColorID 10
  mol modstyle 2 $molid VDW 1.000000 8.000000
  mol modmaterial 2 $molid Opaque
  mol modselect 2 $molid {segname "C.*"}

  #Creates the D dimers
  mol addrep $molid
  mol modcolor 3 $molid ColorID 0
  mol modstyle 3 $molid VDW 1.000000 8.000000
  mol modmaterial 3 $molid Opaque
  mol modselect 3 $molid {segname "D.*"}

  #Creates the bonds  
  mol addrep $molid
  mol modcolor 4 $molid ColorID 1
  mol modstyle 4 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 4 $molid Opaque
  mol modselect 4 $molid {chain A}

  mol addrep $molid
  mol modcolor 5 $molid ColorID 8
  mol modstyle 5 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 5 $molid Opaque
  mol modselect 5 $molid {chain B}

  mol addrep $molid
  mol modcolor 6 $molid ColorID 10
  mol modstyle 6 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 6 $molid Opaque
  mol modselect 6 $molid {chain C}

  mol addrep $molid
  mol modcolor 7 $molid ColorID 0
  mol modstyle 7 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 7 $molid Opaque
  mol modselect 7 $molid {chain D}

  animate delete beg 0 end 0 $molid
}

############################################################
# Loads just a pdb for visualization
############################################################
proc load_pdb {myfile {is_angled 1} {thescale 0.1}} {
  if { [catch {open $myfile r} fid] } {
    puts stderr "File does not exist.\n"
  }
  set molid [mol new $myfile waitfor -1 autobonds off]
  color Display Background white
  
  #mol selection all
  mol delrep 0 $molid

  pbc wrap -center origin -all
  pbc box -color black -center origin
  #mol selection all
  mol delrep 0 $molid

  #Creates the A dimers
  mol addrep $molid
  mol modcolor 0 $molid ColorID 1
  mol modstyle 0 $molid VDW 1.000000 8.000000
  mol modmaterial 0 $molid Opaque
  mol modselect 0 $molid {segname "A.*"}

  #Creates the B dimers
  mol addrep $molid
  mol modcolor 1 $molid ColorID 8
  mol modstyle 1 $molid VDW 1.000000 8.000000
  mol modmaterial 1 $molid Opaque
  mol modselect 1 $molid {segname "B.*"}

  #Creates the C dimers
  mol addrep $molid
  mol modcolor 2 $molid ColorID 10
  mol modstyle 2 $molid VDW 1.000000 8.000000
  mol modmaterial 2 $molid Opaque
  mol modselect 2 $molid {segname "C.*"}

  #Creates the D dimers
  mol addrep $molid
  mol modcolor 3 $molid ColorID 0
  mol modstyle 3 $molid VDW 1.000000 8.000000
  mol modmaterial 3 $molid Opaque
  mol modselect 3 $molid {segname "D.*"}

  #Creates the bonds  
  mol addrep $molid
  mol modcolor 4 $molid ColorID 1
  mol modstyle 4 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 4 $molid Opaque
  mol modselect 4 $molid {chain A}

  mol addrep $molid
  mol modcolor 5 $molid ColorID 8
  mol modstyle 5 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 5 $molid Opaque
  mol modselect 5 $molid {chain B}

  mol addrep $molid
  mol modcolor 6 $molid ColorID 10
  mol modstyle 6 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 6 $molid Opaque
  mol modselect 6 $molid {chain C}

  mol addrep $molid
  mol modcolor 7 $molid ColorID 0
  mol modstyle 7 $molid DynamicBonds 7.0 0.300000 4.000000
  mol modmaterial 7 $molid Opaque
  mol modselect 7 $molid {chain D}
}

proc show_B-C_contacts {molid} {
  mol addrep $molid

  # ALA 36 PHE 18 (orange color)
  set repid [expr [molinfo $molid get numreps] - 1]
  mol modcolor $repid $molid ColorID 3
  mol modstyle $repid $molid VDW 1.500000 8.000000
  mol modmaterial $repid $molid Opaque
  mol modselect $repid $molid (resid 36 and chain B) or (resid 18 and chain C)

  # ASP 29 ARG 127 (yellow color)
  set repid [expr [molinfo $molid get numreps] - 1]
  mol modcolor $repid $molid ColorID 4
  mol modstyle $repid $molid VDW 1.500000 8.000000
  mol modmaterial $repid $molid Opaque
  mol modselect $repid $molid (resid 29 and chain B) or (resid 127 and chain C)

  # LEU 37 VAL 120 (green color)
  set repid [expr [molinfo $molid get numreps] - 1]
  mol modcolor $repid $molid ColorID 7
  mol modstyle $repid $molid VDW 1.500000 8.000000
  mol modmaterial $repid $molid Opaque
  mol modselect $repid $molid (resid 37 and chain B) or (resid 120 and chain C)

  # PHE 23 TYR 132 (pink color)
  set repid [expr [molinfo $molid get numreps] - 1]
  mol modcolor $repid $molid ColorID 9
  mol modstyle $repid $molid VDW 1.500000 8.000000
  mol modmaterial $repid $molid Opaque
  mol modselect $repid $molid (resid 23 and chain B) or (resid 132 and chain C)

  # ASP 32 ARG 127 (purple color)
  set repid [expr [molinfo $molid get numreps] - 1]
  mol modcolor $repid $molid ColorID 11
  mol modstyle $repid $molid VDW 1.500000 8.000000
  mol modmaterial $repid $molid Opaque
  mol modselect $repid $molid (resid 32 and chain B) or (resid 127 and chain C)

  # ILE 139 PRO 134 (lime color)
  set repid [expr [molinfo $molid get numreps] - 1]
  mol modcolor $repid $molid ColorID 12
  mol modstyle $repid $molid VDW 1.500000 8.000000
  mol modmaterial $repid $molid Opaque
  mol modselect $repid $molid (resid 139 and chain B) or (resid 134 and chain C)

  # PRO 25 PRO 129 (yellow color)
  set repid [expr [molinfo $molid get numreps] - 1]
  mol modcolor $repid $molid ColorID 4
  mol modstyle $repid $molid VDW 1.500000 8.000000
  mol modmaterial $repid $molid Opaque
  mol modselect $repid $molid (resid 29 and chain B) or (resid 127 and chain C)


}

############################################################
# RMSD functions
############################################################

# Calculates the RMSD between two specified frames
proc frame_rmsd {selection frame1 frame2} {
  set mol [$selection molindex]
  # check the range
  set num [molinfo $mol get numframes]
  if {$frame1 < 0 || $frame1 >= $num || $frame2 < 0 || $frame2 >= $num} {
    error "frame_rmsd: frame number out of range"
  }
  # get the first coordinate set
  set sel1 [atomselect $mol [$selection text] frame $frame1]
  set coords1 [$sel1 get {x y z}]
  # get the second coordinate set
  set sel2 [atomselect $mol [$selection text] frame $frame2]
  set coords2 [$sel2 get {x y z}]
  # and compute the rmsd values
  set rmsd 0
  foreach coord1 $coords1 coord2 $coords2 {
    set rmsd [expr $rmsd + [veclength2 [vecsub $coord2 $coord1]]]
  }
  # divide by the number of atoms and return the result
  return [expr sqrt($rmsd / ([$selection num] + 0.0))]
}

# Aligns all frames to frame 0 to remove COM drift
proc align_frames {mol_id} {
  set ref [atomselect $mol_id all frame 0]
  set sel [atomselect $mol_id all]
  for {set i 0} {$i < [molinfo $mol_id get numframes]} {incr i} {
    $sel frame $i
    $sel move [measure fit $sel $ref]
  }
}

# Calculates the RMSD of the top loaded molecule
proc get_rmsd {mol_id} {
  puts "Getting RMSD of mol_id: $mol_id"
  align_frames $mol_id
  set sel [atomselect $mol_id all]

  # Outputs to rmsd.out
  set outfile1 [open "rmsd.out" w+]

  for {set i 0} {$i < [molinfo $mol_id get numframes]} {incr i} {
    puts $outfile1 [list $i [frame_rmsd $sel $i 0]]
  }
  close $outfile1
}

############################################################
# Creates a folder of img files to make a movie out of with ffmpeg
############################################################
proc make_movie {path interv maxframe} {

  animate goto start
  set i [molinfo top get frame]
  set stop $maxframe ;#[molinfo top get numframes]

  while {$i < [expr $stop + 1]} {

    file mkdir $path ;# [file mkdir] in Tcl is like mkdir -p
    file mkdir "$path/images"
    #render Tachyon [format "$path/images/image_%05d" $i] "/usr/local/lib/vmd/tachyon_LINUXAMD64" -aasamples 12 %s -format TARGA -res 2000 2000 -o %s.tga
    render TachyonLOptiXInternal [format "$path/images/image_%05d.tga" $i] -res 2000 2000 %s
    #render Tachyon [format "$path/images/image_%05d" $i] "/Applications/VMD 1.9.4a57-arm64-Rev12.app/Contents/vmd/tachyon_MACOSXARM64" -aasamples 12 %s -format TARGA -res 2000 2000 -o %s.tga
          render TachyonLOptiXInternal [format "$path/$mol_id/images/image_%05d.tga" $i] -res 2000 2000 %s
    set i [expr $i + $interv]
    animate goto $i
  }

}

proc do_ffmpeg {folder framerate} {
  set foldername [file tail $folder]
  set outfile "$folder/${foldername}_video.mp4"

  set cmd "ffmpeg -r $framerate -pattern_type glob -y -i \"${folder}/images/*.tga\" -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -vcodec libx264 -crf 18 -pix_fmt yuv444p \"$outfile\""

  if {[catch {eval exec -ignorestderr $cmd 2>@1} out]} {
    echo error "'ffmpeg' execution command failed."
    echo debug "reason= $out"
  } else {
    puts "Video saved to: $outfile"
  }
}

############################################################
# Loads a pdb/trajectory, aligns every frame to the starting
# frame (frame 0) to minimize RMSD from the starting structure,
# then renders and encodes a video of the aligned trajectory.
# Output: <path>/<foldername>_video.mp4
# Usage: align_and_render_video <pdb_file> <trajectory_file> <path> [stride=1] [interv=1] [framerate=30]
############################################################
proc align_and_render_video {pdb_file trajectory_file path {stride 1} {interv 1} {framerate 30}} {
  load_traj $pdb_file $trajectory_file $stride
  set molid [molinfo top get id]

  # Aligns all frames to frame 0, minimizing RMSD from the starting frame
  align_frames $molid

  set maxframe [expr [molinfo $molid get numframes] - 1]
  make_and_render_video $path $interv $maxframe $framerate
}

############################################################
# Renders frames and encodes a video in one step.
# Output: <path>/<foldername>_video.mp4
# Usage: make_and_render_video <path> <interv> <maxframe> [framerate=1]
############################################################
proc make_and_render_video {path interv maxframe {framerate 30}} {
  set imgdir "$path/images"
  file mkdir $imgdir

  animate goto start
  set i [molinfo top get frame]

  while {$i < [expr $maxframe + 1]} {
    render TachyonLOptiXInternal [format "$imgdir/image_%05d.tga" $i] -res 2000 2000 %s
    set i [expr $i + $interv]
    animate goto $i
  }

  set foldername [file tail $path]
  set outfile "$path/${foldername}_video.mp4"

  set cmd "ffmpeg -r $framerate -pattern_type glob -y -i \"${imgdir}/*.tga\" -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -vcodec libx264 -crf 18 -pix_fmt yuv444p \"$outfile\""

  if {[catch {eval exec -ignorestderr $cmd 2>@1} out]} {
    echo error "'ffmpeg' execution command failed."
    echo debug "reason= $out"
  } else {
    puts "Video saved to: $outfile"
  }
}