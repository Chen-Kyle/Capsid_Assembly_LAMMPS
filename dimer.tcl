############################################################
# Loads a pdb file and trajectory file
############################################################
proc load_traj {pdb_file trajectory_file} {
  if { [catch {open $pdb_file r} fid] } {
    puts stderr "File does not exist.\n"
  }

  #Loads in pdb and trajectory files
  set molid [mol new $pdb_file waitfor all]
  mol addfile $trajectory_file waitfor all

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
  mol modstyle 0 $molid VDW 1.000000 12.000000
  mol modmaterial 0 $molid Opaque
  mol modselect 0 $molid {segname "A.*"}

  #Creates the B dimers
  mol addrep $molid
  mol modcolor 1 $molid ColorID 8
  mol modstyle 1 $molid VDW 1.000000 12.000000
  mol modmaterial 1 $molid Opaque
  mol modselect 1 $molid {segname "B.*"}

  #Creates the C dimers
  mol addrep $molid
  mol modcolor 2 $molid ColorID 10
  mol modstyle 2 $molid VDW 1.000000 12.000000
  mol modmaterial 2 $molid Opaque
  mol modselect 2 $molid {segname "C.*"}

  #Creates the D dimers
  mol addrep $molid
  mol modcolor 3 $molid ColorID 0
  mol modstyle 3 $molid VDW 1.000000 12.000000
  mol modmaterial 3 $molid Opaque
  mol modselect 3 $molid {segname "D.*"}

  #Creates the bonds  
  mol addrep $molid
  mol modcolor 4 $molid ColorID 1
  mol modstyle 4 $molid DynamicBonds 7.0 0.300000 5.000000
  mol modmaterial 4 $molid Opaque
  mol modselect 4 $molid {chain A}

  mol addrep $molid
  mol modcolor 5 $molid ColorID 8
  mol modstyle 5 $molid DynamicBonds 7.0 0.300000 5.000000
  mol modmaterial 5 $molid Opaque
  mol modselect 5 $molid {chain B}

  mol addrep $molid
  mol modcolor 6 $molid ColorID 10
  mol modstyle 6 $molid DynamicBonds 7.0 0.300000 5.000000
  mol modmaterial 6 $molid Opaque
  mol modselect 6 $molid {chain C}

  mol addrep $molid
  mol modcolor 7 $molid ColorID 0
  mol modstyle 7 $molid DynamicBonds 7.0 0.300000 5.000000
  mol modmaterial 7 $molid Opaque
  mol modselect 7 $molid {chain D}
}

############################################################
# Loads just a pdb for visualization
############################################################
proc load_pdb {myfile {is_angled 1} {thescale 0.1}} {
  if { [catch {open $myfile r} fid] } {
    puts stderr "File does not exist.\n"
  }
  set molid [mol new $myfile waitfor -1 autobonds off]

  #mol selection all
  mol delrep 0 $molid

  #Creates the A dimers
  mol addrep $molid
  mol modcolor 0 $molid ColorID 1
  mol modstyle 0 $molid Tube 0.300000 12.000000
  mol modmaterial 0 $molid Opaque
  mol modselect 0 $molid {segname "A.*"}

  #Creates the B dimers
  mol addrep $molid
  mol modcolor 1 $molid ColorID 8
  mol modstyle 1 $molid Tube 0.300000 12.000000
  mol modmaterial 1 $molid Opaque
  mol modselect 1 $molid {segname "B.*"}

  #Creates the C dimers
  mol addrep $molid
  mol modcolor 2 $molid ColorID 10
  mol modstyle 2 $molid Tube 0.300000 12.000000
  mol modmaterial 2 $molid Opaque
  mol modselect 2 $molid {segname "C.*"}

  #Creates the D dimers
  mol addrep $molid
  mol modcolor 3 $molid ColorID 0
  mol modstyle 3 $molid Tube 0.300000 12.000000
  mol modmaterial 3 $molid Opaque
  mol modselect 3 $molid {segname "D.*"}

  #Draws a sphere at the origin
  draw sphere {0 0 0} radius 1.0
  # pbc box -color black -center origin
  # pbc wrap -all -center origin
  # animate goto start
}

############################################################
# Calculates the RMSD between two specified frames
############################################################

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

############################################################
# Aligns all frames to frame 0 to remove COM drift
############################################################

proc align_frames {mol_id} {
  set ref [atomselect $mol_id all frame 0]
  set sel [atomselect $mol_id all]
  for {set i 0} {$i < [molinfo $mol_id get numframes]} {incr i} {
    $sel frame $i
    $sel move [measure fit $sel $ref]
  }
}

############################################################
# Calculates the RMSD of the top loaded molecule
############################################################

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
