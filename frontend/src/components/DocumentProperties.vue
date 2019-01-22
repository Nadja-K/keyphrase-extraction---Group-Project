<template>
    <div>
        <v-card class="pa-4">
            <h6 class="title mb-2">Common Properties</h6>
            <v-checkbox
                v-model="showCandidateKeyphrases"
                label="candidate keyphrases"
                hide-details
                class="mt-1"
            />
            <v-checkbox
                v-model="showSelectedKeyphrases"
                label="selected keyphrases"
                hide-details
                class="mt-1"
            />
        </v-card>
        <v-card class="pa-4 mt-3">
            <!-- <h6 class="title mb-2">KeyCluster Properties</h6>
            <v-checkbox
                v-model="allSelected"
                :indeterminate="allSelectedIndetermined"
                label="select all"
                hide-details
                class="mt-1"
            />
            <v-divider />
            <div :style="{overflowY:'scroll', maxHeight: '180px'}">
                <v-checkbox v-for="clusterId in selectedRun.num_clusters"
                    v-model="selectedClusters"
                    :label="`cluster ${clusterId}`"
                    hide-details
                    class="mt-1"
                    :color="`#${palette[clusterId]}`"
                    :id="`cluster ${clusterId}`"
                    :value="clusterId"
                    :key="clusterId"
                />
            </div> -->
            <PropertyList :items="items"/>
        </v-card>
    </div>
</template>

<script>
import { mapActions, mapState, mapGetters } from 'vuex';
import palette from '../palette';
import PropertyList from './PropertyList.vue';

export default {
    components: {
        PropertyList,
    },
    computed: {
        ...mapState('document', [
            'document',
            'ui',
        ]),
        ...mapGetters('document', [
            'selectedRun',
        ]),
        showCandidateKeyphrases: {
            get() {
                return this.ui.commonProperties.showCandidateKeyphrases;
            },
            set(value) {
                this.setCommonProperty({ showCandidateKeyphrases: value });
            },
        },
        showSelectedKeyphrases: {
            get() {
                return this.ui.commonProperties.showSelectedKeyphrases;
            },
            set(value) {
                this.setCommonProperty({ showSelectedKeyphrases: value });
            },
        },
        selectedClusters: {
            get() {
                return this.ui.keyClusterProperties.selectedClusters;
            },
            set(value) {
                if (value.length > 0 && value.length < this.selectedRun.num_clusters) {
                    this.allSelectedIndetermined = true;
                    this.allSelectedState = false;
                } else if (value.length === this.selectedRun.num_clusters) {
                    this.allSelectedIndetermined = false;
                    this.allSelectedState = true;
                } else {
                    this.allSelectedIndetermined = false;
                }
                this.setKeyClusterProperty({ selectedClusters: value });
            },
        },
        allSelected: {
            get() {
                return this.allSelectedState;
            },
            set(value) {
                let allClusters = [];
                if (value && !this.allSelectedIndetermined) {
                    allClusters = [...Array(this.selectedRun.num_clusters + 1).keys()].slice(1);
                }
                this.setKeyClusterProperty({ selectedClusters: allClusters });

                this.allSelectedState = !this.allSelectedIndetermined;
                this.allSelectedIndetermined = false;
            },
        },
        items() {
            return [...Array(this.selectedRun.num_clusters)].map((x, i) => ({
                name: `cluster ${i}`,
            }));
        },
    },
    data() {
        return {
            palette,
            allSelectedState: false,
            allSelectedIndetermined: false,
        };
    },
    methods: {
        ...mapActions('document/ui', [
            'setCommonProperty',
            'setKeyClusterProperty',
        ]),
    },
};
</script>

<style>

</style>
